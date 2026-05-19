import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

ImageSize = 28

InputSize = 784
HiddenSize = 128
OutputSize = 10

Epochs = 5000
Lr = 0.1

X = []
Y = []


def Convert_To_GrayScale(Image):
    return cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)


def Apply_Threshold(Image):
    l, Threshold = cv2.threshold(Image, 127, 255, cv2.THRESH_BINARY_INV)
    return Threshold


def Resize_Image(Image):
    return cv2.resize(Image, (ImageSize, ImageSize))


def Normalize_Image(Image):
    return Image / 255.0


def Flatten_Image(Image):
    return Image.flatten()


def Preprocess_Image(Image):
    Gray = Convert_To_GrayScale(Image)
    Threshold = Apply_Threshold(Gray)
    Resized = Resize_Image(Threshold)
    Normalized = Normalize_Image(Resized)
    Flattened = Flatten_Image(Normalized)
    return Flattened


def Load_Dataset():
    global X
    global Y
    
    plt.figure(figsize=(10, 5))
    for Digit in range(10):
        FolderPath = os.path.join("dataset", str(Digit))
        
        for ImageIndex in range(24):
            ImagePath = os.path.join(FolderPath, f"{ImageIndex}.png")
            Image = cv2.imread(ImagePath)

            if Image is None:
                continue

            if ImageIndex == 0:
                RGB = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)
                plt.subplot(2, 5, Digit + 1)
                plt.imshow(RGB)
                plt.title(f"Digit {Digit}")
                plt.axis("off")
            
            Processed = Preprocess_Image(Image)
            X.append(Processed)
            Y.append(Digit)
    plt.show()
    X_Array = np.array(X)
    Y_Array = np.array(Y)
    return X_Array, Y_Array


def One_Hot_Encode(Y):
    Encoded = np.zeros((Y.size, 10))
    Encoded[np.arange(Y.size), Y] = 1
    return Encoded


def Shuffle_Data(X, Y):
    Indices = np.arange(X.shape[0])
    np.random.shuffle(Indices)
    return X[Indices], Y[Indices]


def Split_Data(X, Y):

    SplitIndex = int(0.8 * X.shape[0])
    X_Train = X[:SplitIndex]
    Y_Train = Y[:SplitIndex]
    X_Test = X[SplitIndex:]
    Y_Test = Y[SplitIndex:]

    return X_Train, Y_Train, X_Test, Y_Test


def ReLU(Z):
    return np.maximum(0, Z)


def ReLU_Derivative(Z):
    return Z > 0


def Softmax(Z):
    Exp = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return Exp / np.sum(Exp, axis=1, keepdims=True)


def Initialize_Parameters():

    W1 = np.random.randn(InputSize, HiddenSize) * 0.01
    b1 = np.zeros((1, HiddenSize))
    W2 = np.random.randn(HiddenSize, OutputSize) * 0.01
    b2 = np.zeros((1, OutputSize))
    return W1, b1, W2, b2


def Forward_Backward(X, Y, W1, b1, W2, b2):

    Z1 = np.dot(X, W1) + b1
    A1 = ReLU(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = Softmax(Z2)
    Loss = -np.mean(np.sum(Y * np.log(A2 + 1e-8), axis=1))
    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2) / X.shape[0]
    db2 = np.sum(dZ2, axis=0, keepdims=True) / X.shape[0]
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * ReLU_Derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / X.shape[0]
    db1 = np.sum(dZ1, axis=0, keepdims=True) / X.shape[0]

    return Loss, dW1, db1, dW2, db2


def Predict(X, W1, b1, W2, b2):

    Z1 = np.dot(X, W1) + b1
    A1 = ReLU(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = Softmax(Z2)

    return np.argmax(A2, axis=1)


def Train_Model(X_Train, Y_Train):

    W1, b1, W2, b2 = Initialize_Parameters()
    Losses = []
    for Epoch in range(Epochs):

        Loss, dW1, db1, dW2, db2 = Forward_Backward(X_Train,Y_Train,
            W1,b1,W2,b2)

        W1 = W1 - (Lr * dW1)
        b1 = b1 - (Lr * db1)
        W2 = W2 - (Lr * dW2)
        b2 = b2 - (Lr * db2)
        Losses.append(Loss)
        if Epoch % 100 == 0:
            print("Epoch:", Epoch, "Loss:", Loss)

    plt.plot(Losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.show()

    return W1, b1, W2, b2


def Evaluate_Model(X_Test, Y_Test, W1, b1, W2, b2):
    Predictions = Predict(X_Test, W1, b1, W2, b2)
    Labels = np.argmax(Y_Test, axis=1)
    for i in range(len(Predictions)):
        print(f"Predicted : {Predictions[i]} , Actual: {Labels[i]}")
    Accuracy = np.mean(Predictions == Labels) * 100
    print("Accuracy:", Accuracy)


def Find_Contours(Image):

    Contours, _ = cv2.findContours(Image,cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

    return Contours


def Extract_Digits(Image, Contours):
    Digits = []
    Boxes = []

    for Contour in Contours:

        X, Y, W, H = cv2.boundingRect(Contour)
        if W > 10 and H > 10:
            Boxes.append((X, Y, W, H))

    Boxes = sorted(Boxes, key=lambda Box: Box[0])

    for Box in Boxes:
        X, Y, W, H = Box

        Digit = Image[Y:Y + H, X:X + W]
        Digit = cv2.resize(Digit, (28, 28))
        Digit = Digit / 255.0
        Digit = Digit.flatten()

        Digits.append(Digit)

    return Digits, Boxes


def Predict_Multiple_Digits(Image, W1, b1, W2, b2):

    Gray = Convert_To_GrayScale(Image)
    Threshold = Apply_Threshold(Gray)
    Contours = Find_Contours(Threshold)
    Digits, Boxes = Extract_Digits(Threshold, Contours)
    Result = ""

    for Digit in Digits:
        Prediction = Predict(
            np.array([Digit]),W1,b1,W2,b2)[0]
        Result += str(Prediction)

    for Box in Boxes:
        X, Y, W, H = Box
        cv2.rectangle(Image, (X, Y), (X + W, Y + H), (0, 255, 0), 2)

    cv2.putText(Image,Result,(10, 40),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),2)
    return Image



X, Y = Load_Dataset()
Y = One_Hot_Encode(Y)
X, Y = Shuffle_Data(X, Y)
X_Train, Y_Train, X_Test, Y_Test = Split_Data(X, Y)
W1, b1, W2, b2 = Train_Model(X_Train, Y_Train)
Evaluate_Model(X_Test, Y_Test, W1, b1, W2, b2)