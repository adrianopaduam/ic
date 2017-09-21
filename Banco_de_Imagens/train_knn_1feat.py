
import csv
import os
import pdb # package for debugging
import sys

import nibabel as nib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def extract_feat_vector_1D(img, x, y, z):
    return [img[x,y,z]]



def main():
    if len(sys.argv) != 5:
        sys.exit("python train_knn_1feat.py <ref_img_orig.hdr> "
                 "<markers_ref_img.csv> <k> <out_knn>")

    ref_img_path = sys.argv[1]
    markers_path = sys.argv[2]
    k            = int(sys.argv[3])
    out_knn_path = sys.argv[4]

    print("Reference Image: %s" % ref_img_path)
    print("Markers CSV: %s" % markers_path)
    print("Output KNN file: %s\n" % out_knn_path)

    # creates the parent directory for the output knn file
    parent_dir = os.path.dirname(out_knn_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    print("- Reading Reference Image")
    ref_img      = nib.load(ref_img_path)
    ref_img_data = ref_img.get_data()
    ref_img_data = ref_img_data[:,:,:,0]

    print("- Reading samples from CSV")
    train_set = [] # matrix of features - each row is a feature vector from a
    # sample
    labels = [] # label from the samples - each row [i] corresponds to the label
    # from the sample [i]
    with open(markers_path, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            # convert all string elements from the row to integer values
            x, y, z, label = map(int, row)

            # create a feature vector (list) for the sample with only the
            # brightness/value from the voxel with coordinates [x][y][z]
            feat_vector = extract_feat_vector_1D(ref_img_data, x, y, z)
            train_set.append(feat_vector) # append the feature vector into training set
            labels.append(label)
    train_set = np.array(train_set)
    labels = np.array(labels)


    print("- Training kNN classifier with k = %d" % k)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_set, labels)

    print("- Writing the kNN Classifier")

    print("- Classifying the Reference Image")
    print("- Extracting Feature Vectors for each voxel (testing sample) from "
          "the Ref Image")
    test_set = []
    for x in range(ref_img_data.shape[0]):
        for y in range(ref_img_data.shape[1]):
            for z in range(ref_img_data.shape[2]):
                feat_vector = extract_feat_vector_1D(ref_img_data, x, y, z)
                test_set.append(feat_vector)
    test_set = np.array(test_set)
    print("- Predicting the testing samples")
    labeled_data = knn.predict(test_set)

    print("- Reshape Output Labeled Image")
    labeled_data = labeled_data.reshape(ref_img_data.shape)
    data = nib.analyze.AnalyzeImage(labeled_data, np.eye(4),
                                    header=ref_img.get_header())
    nib.save(data, out_knn_path)

    print("Done...")

if __name__ == "__main__":
    main()

