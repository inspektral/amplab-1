import essentia
import essentia.standard as es
import numpy as np
import json

essentia.log.warningActive = False

def load_classes(json_path):
    with open(json_path, 'r') as f:
        data =  json.load(f)
        return data["classes"]


class Analyzer:

    def __init__(self, fs=16000):
        self.mono_mixer = es.MonoMixer()
        self.resample = es.Resample(outputSampleRate=fs, quality=0)
        self.loudness = es.LoudnessEBUR128()
        self.rhythm_extractor = es.RhythmExtractor2013()

        self.key_extractor_temperley = es.KeyExtractor(sampleRate=fs, profileType='temperley')
        self.key_extractor_krumhansl = es.KeyExtractor(sampleRate=fs, profileType='krumhansl')
        self.key_extractor_edma = es.KeyExtractor(sampleRate=fs, profileType='edma')
        
        self.discogs_embeddings = es.TensorflowPredictEffnetDiscogs(graphFilename="./weights/discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
        
        self.discogs_style = es.TensorflowPredict2D(graphFilename="./weights/genre_discogs400-discogs-effnet-1.pb", input="serving_default_model_Placeholder", output="PartitionedCall:0")
        self.discogs_classes = load_classes("./model_metadata/genre_discogs400-discogs-effnet-1.json")

        self.discogs_instrumental = es.TensorflowPredict2D(graphFilename="./weights/voice_instrumental-discogs-effnet-1.pb", output="model/Softmax")
        self.discogs_instrumental_classes = load_classes("./model_metadata/voice_instrumental-discogs-effnet-1.json")


    def analyze(self, audio_file_path):
        results = {}

        audio, _, _, _, _, _ = es.AudioLoader(filename=audio_file_path)()
        _, _, results["loudness"], _ = self.loudness(audio)

        mono_audio = self.resample(self.mono_mixer(audio, 2))

        results["bpm"], _, _, _, _ = self.rhythm_extractor(mono_audio)

        results["key_temperley"], results["scale_temperley"], _ = self.key_extractor_temperley(mono_audio)
        results["key_krumhansl"], results["scale_krumhansl"], _ = self.key_extractor_krumhansl(mono_audio)
        results["key_edma"], results["scale_edma"], _ = self.key_extractor_edma(mono_audio)

        discogs_embeddings = self.discogs_embeddings(mono_audio)
        results["discogs_embeddings_mean"] = np.mean(discogs_embeddings, axis=0).tolist()

        style = np.mean(self.discogs_style(discogs_embeddings), axis=0)
        style_class_index = int(np.argmax(style))
        results["style"] = self.discogs_classes[style_class_index]

        instrumental = np.mean(self.discogs_instrumental(discogs_embeddings), axis=0)
        instrumental_class_index = int(np.argmax(instrumental))
        results["instrumental"] = self.discogs_instrumental_classes[instrumental_class_index]

        return results
