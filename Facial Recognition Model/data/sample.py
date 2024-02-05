import pandas as pd
import numpy as np
'''This program generates a csv file with the appropriate labels'''

# Generating sample data (random numbers here, replace this with actual data)
num_samples = 1  # Number of samples
num_features = 63   # Number of facial landmarks

# Creating random data for each facial landmark position
data = np.random.rand(num_samples, num_features)

# Creating column names for each facial landmark position
columns = [
    "BrowLowererL", "BrowLowererR", "CheekPuffL", "CheekPuffR", "CheekRaiserL", "CheekRaiserR",
    "CheekSuckL", "CheekSuckR", "ChinRaiserB", "ChinRaiserT", "DimplerL", "DimplerR",
    "EyesClosedL", "EyesClosedR", "EyesLookDownL", "EyesLookDownR", "EyesLookLeftL", "EyesLookLeftR",
    "EyesLookRightL", "EyesLookRightR", "EyesLookUpL", "EyesLookUpR", "InnerBrowRaiserL", "InnerBrowRaiserR",
    "JawDrop", "JawSidewaysLeft", "JawSidewaysRight", "JawThrust", "LidTightenerL", "LidTightenerR",
    "LipCornerDepressorL", "LipCornerDepressorR", "LipCornerPullerL", "LipCornerPullerR", "LipFunnelerLB",
    "LipFunnelerLT", "LipFunnelerRB", "LipFunnelerRT", "LipPressorL", "LipPressorR", "LipPuckerL", "LipPuckerR",
    "LipStretcherL", "LipStretcherR", "LipSuckLB", "LipSuckLT", "LipSuckRB", "LipSuckRT", "LipTightenerL",
    "LipTightenerR", "LipsToward", "LowerLipDepressorL", "LowerLipDepressorR", "MouthLeft", "MouthRight",
    "NoseWrinklerL", "NoseWrinklerR", "OuterBrowRaiserL", "OuterBrowRaiserR", "UpperLidRaiserL", "UpperLidRaiserR",
    "UpperLipRaiserL", "UpperLipRaiserR"
]


# Adding the generated data to a pandas DataFrame
df = pd.DataFrame(data, columns=columns)

# Adding a sample emotion column with random emotions (for demonstration)
emotions = ["Happy", "Sad", "Mad", "Anxious"]
df['Emotion'] = np.random.choice(emotions, num_samples)

# Save the generated data to a CSV file
df.to_csv('sample_single_emotion.csv', index=False)
