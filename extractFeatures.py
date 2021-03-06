
import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

DATASET = "./datasetGTZAN"


def extractFeatures(path, SR=22050, HOP_LEN = 256, FRAME_LEN = 512):
    features = [path]
    x, _ = librosa.load(path, sr=SR)
    
    rmse = librosa.feature.rms(x, frame_length=FRAME_LEN, hop_length=HOP_LEN)[0]
    features.extend([np.mean(rmse), np.std(rmse)])
    
    zeroCrossRate = librosa.feature.zero_crossing_rate(x, frame_length=FRAME_LEN, hop_length=HOP_LEN)[0]
    features.extend([np.mean(zeroCrossRate), np.std(zeroCrossRate)])
    
    tempo = librosa.beat.tempo(x, sr=SR, hop_length=HOP_LEN)[0]
    features.append(tempo)
    
    spectralCentroid = librosa.feature.spectral_centroid(x, sr=SR, hop_length=HOP_LEN, n_fft=FRAME_LEN)[0]
    features.extend([np.mean(spectralCentroid), np.std(spectralCentroid)])
    
    spectralBandwith = librosa.feature.spectral_bandwidth(x, sr=SR, hop_length=HOP_LEN, n_fft=FRAME_LEN)[0]
    features.extend([np.mean(spectralBandwith), np.std(spectralBandwith)])
    
    spectralRolloff = librosa.feature.spectral_rolloff(x, sr=SR, hop_length=HOP_LEN, n_fft=FRAME_LEN)[0]
    features.extend([np.mean(spectralRolloff), np.std(spectralRolloff)])
    
    spectralContrast = librosa.feature.spectral_contrast(x, sr=SR, hop_length=HOP_LEN, n_fft=FRAME_LEN)
    for i in range(spectralContrast.shape[0]):
        features.extend([np.mean(spectralContrast[i]), np.std(spectralContrast[i])])
    
    chromagram = librosa.feature.chroma_stft(x, sr=SR, hop_length=HOP_LEN, n_fft=FRAME_LEN)
    for i in range(chromagram.shape[0]):
        features.extend([np.mean(chromagram[i]), np.std(chromagram[i])])

    mfcc = librosa.feature.mfcc(x, sr=SR)
    for i in range(mfcc.shape[0]):
        features.extend([np.mean(mfcc[i]), np.std(mfcc[i])])

    return features


def extractImages(pathList, SR=22050, HOP_LEN = 256, FRAME_LEN = 512):
    path, songName = pathList
    x, _ = librosa.load(path, sr=SR)

    # Spectrogram
    stft = librosa.stft(x, hop_length=HOP_LEN)
    stft_db = librosa.amplitude_to_db(abs(stft), ref=np.max)
    _, ax = plt.subplots(dpi=128)
    librosa.display.specshow(stft_db, y_axis="log", sr=SR)
    ax.set_axis_off()
    plt.savefig(os.path.join("./GTZAN/spectrogram",songName+".png"),bbox_inches='tight', pad_inches=0)

    step = stft_db.shape[1]//10
    clipsList = [stft_db[:,i:i+step] for i in range(0, stft_db.shape[1], step)]
    clipsList.pop()
    for i, clip in enumerate(clipsList):
        _, ax = plt.subplots(dpi=128)
        librosa.display.specshow(clip, y_axis="mel", sr=SR)
        ax.set_axis_off()
        plt.savefig(os.path.join("./GTZAN/spectrogram3s",f"{songName}.{i}.png"),bbox_inches='tight', pad_inches=0)

    # Mel-Spectrogram
    melStft = librosa.feature.melspectrogram(x, sr=SR, hop_length=HOP_LEN)
    melStft_db = librosa.amplitude_to_db(melStft, ref=np.max)
    _, ax = plt.subplots(dpi=128)
    librosa.display.specshow(melStft_db, y_axis="mel", sr=SR)
    ax.set_axis_off()
    plt.savefig(os.path.join("./GTZAN/melSpectrogram",songName+".png"),bbox_inches='tight', pad_inches=0)

    step = melStft_db.shape[1]//10
    clipsList = [melStft_db[:,i:i+step] for i in range(0, melStft_db.shape[1], step)]
    clipsList.pop()
    for i, clip in enumerate(clipsList):
        _, ax = plt.subplots(dpi=128)
        librosa.display.specshow(clip, y_axis="mel", sr=SR)
        ax.set_axis_off()
        plt.savefig(os.path.join("./GTZAN/melSpectrogram3s",f"{songName}.{i}.png"),bbox_inches='tight', pad_inches=0)

    # Mel-MFCC
    mfcc = librosa.feature.mfcc(x, sr=SR, n_mfcc=20)
    mfcc_db = librosa.amplitude_to_db(mfcc, ref=np.max)
    _, ax = plt.subplots(dpi=128)
    librosa.display.specshow(mfcc_db, y_axis="mel", sr=SR)
    ax.set_axis_off()
    plt.savefig(os.path.join("./GTZAN/mfcc",songName+".png"),bbox_inches='tight', pad_inches=0)
    plt.close("all")


if __name__ == '__main__':
    
    columns = ["path", "rmse-mean", "rmse-std", "zero-cross-rate-mean", "zero-cross-rate-std", "tempo", "spectral-centroid-mean", "spectral-centroid-std",
    "spectral-bandwith-mean", "spectral-bandwith-std", "spectral-rolloff-mean", "spectral-rolloff-std", "spectral-contrast1-mean", "spectral-contrast1-std",
    "spectral-contrast2-mean", "spectral-contrast2-std","spectral-contrast3-mean", "spectral-contrast3-std","spectral-contrast4-mean", "spectral-contrast4-std",
    "spectral-contrast5-mean", "spectral-contrast5-std","spectral-contrast6-mean", "spectral-contrast6-std","spectral-contrast7-mean", "spectral-contrast7-std",
    "chroma1-mean", "chroma1-std", "chroma2-mean", "chroma2-std", "chroma3-mean", "chroma3-std", "chroma4-mean", "chroma4-std", "chroma5-mean", "chroma5-std",
    "chroma6-mean", "chroma6-std", "chroma7-mean", "chroma7-std", "chroma8-mean", "chroma8-std", "chroma9-mean", "chroma9-std", "chroma10-mean", "chroma10-std",
    "chroma11-mean", "chroma11-std", "chroma12-mean", "chroma12-std", "mfcc1-mean", "mfcc1-std", "mfcc2-mean", "mfcc2-std", "mfcc3-mean", "mfcc3-std",
    "mfcc4-mean", "mfcc4-std", "mfcc5-mean", "mfcc5-std", "mfcc6-mean", "mfcc6-std", "mfcc7-mean", "mfcc7-std", "mfcc8-mean", "mfcc8-std",
    "mfcc9-mean", "mfcc9-std", "mfcc10-mean", "mfcc10-std", "mfcc11-mean", "mfcc11-std", "mfcc12-mean", "mfcc12-std", "mfcc13-mean", "mfcc13-std",
    "mfcc14-mean", "mfcc14-std", "mfcc15-mean", "mfcc15-std", "mfcc16-mean", "mfcc16-std", "mfcc17-mean", "mfcc17-std", "mfcc18-mean", "mfcc18-std", "mfcc19-mean",
    "mfcc19-std", "mfcc20-mean", "mfcc20-std", "class"]


    # FOR EXTRACTING FEATURES
    # paths = []
    # for i, genre in enumerate(os.listdir(DATASET)):
    #     for song in os.listdir(os.path.join(DATASET, genre)):
    #         songPath = os.path.join(DATASET, genre, song)
    #         paths.append([songPath, i])
    # data = []
    # for path, targetClass in tqdm(paths):
    #     features = extractFeatures(path)
    #     features.append(targetClass)
    #     data.append(features)

    # df = pd.DataFrame(data=data, columns=columns)
    # df.to_pickle("rawFeaturesGTZAN.pkl")


    # FOR EXTRACTING IMAGES
    paths = []
    for i, genre in enumerate(os.listdir(DATASET)):
        for song in os.listdir(os.path.join(DATASET, genre)):
            songPath = os.path.join(DATASET, genre, song)
            songName = ".".join(song.split(".")[:-1])
            paths.append([songPath, songName])
    
    matplotlib.use("Agg")
    for path in tqdm(paths):
        extractImages(path)