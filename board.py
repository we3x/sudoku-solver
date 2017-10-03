import cv2
from model import PCAModel
import numpy as np
import math

class SudokuBoard(object):
    numbers = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

    def __init__(self, path):
        self.path = path
        self.image_core = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        self.board = self.init_board()
        self.data = self.load_training_set()
        self.EigenNumberSudoku = PCAModel()
        self.EigenNumberSudoku.train(self.data)
        self.image_bin = self.prepare_image()
        self.recognize()

    def init_board(self):
        board = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
        return board

    def prepare_image(self):
        image_gs = cv2.cvtColor(self.image_core, cv2.COLOR_RGB2GRAY)
        ret, image_bin = cv2.threshold(image_gs, 130, 255, cv2.THRESH_BINARY)
        img_bin = cv2.resize(image_bin, (270, 270))
        return img_bin

    def load_training_set(self):
        data = []
        for color in self.colors:
            for number in self.numbers:
                    path = './images/numbers/'+number+'.png'
                    label = number
                    component_core = cv2.imread(path)
                    component_gs = cv2.cvtColor(component_core, cv2.COLOR_RGB2GRAY)
                    ret, component_bin = cv2.threshold(component_gs, 130, 255, cv2.THRESH_BINARY)
                    component = cv2.resize(component_bin, (30, 30))
                    vector = np.array(component).ravel()
                    data.append({'label': label, 'sample': vector})
        return data

    def recognize(self):
        for i in range(9):
            for j in range(9):
                vector = np.array(self.image_bin[i*30:(i+1)*30, j*30:(j+1)*30]).ravel()
                label, res = self.EigenNumberSudoku.classify(vector)
                self.board[i][j] = (self.numbers.index(label)+1)


    def get_board(self):
        return self.board
