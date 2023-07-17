# this file is for handling inference process in term of input and output prediction
import numpy as np
import torch

class Inference:
    def __init__(self, model, input_prep):
        self.model = model
        self.input_prep = input_prep

    def infer(self):
        with torch.no_grad():
            # self.model.eval()
            logits = self.model(self.input_prep)
            ps = torch.exp(logits)
            _, predTest = torch.max(ps,1)
            print("done predicting")
            print("ps :",_, "pred :",predTest)
        return predTest