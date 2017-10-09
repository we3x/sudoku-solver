import cv2
from model import PCAModel
from sudoku import soduko
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
        self.clear_line(0,0)
        self.recognize()
        self.str_board = self.prepare_board()
        self.t = soduko(self.str_board)
        self.t.one_level_supposition()
        self.t.check()
        self.board = self.str_to_int(self.t.as_test_list())

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

    def prepare_board(self):
        str_board = ["","","","","","","","",""]
        for i in range(9):
            for j in range(9):
                str_board[i] += str(self.board[i][j])
        return str_board

    def str_to_int(self, str_board):
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
        for i in range(9):
            for j in range(9):
                board[i][j] = int(str_board[i][j])
        return board

    def clear_line(self,a,b):
        theStack = [ (a, b) ]
        while len(theStack) > 0:
            x, y = theStack.pop()
            if self.image_bin[x][y] == 255:
                continue
            self.image_bin[x][y] = 255
            if (x < 269):
                theStack.append( (x + 1, y) )
            if (x > 0):
                theStack.append( (x - 1, y) )
            if (y < 269):
                theStack.append( (x, y + 1) )
            if (y > 0):
                theStack.append( (x, y - 1) )

    def prepare_image(self):
        image_gs = cv2.cvtColor(self.image_core, cv2.COLOR_RGB2GRAY)
        ret, image_bin = cv2.threshold(image_gs, 130, 255, cv2.THRESH_BINARY)
        img_bin = cv2.resize(image_bin, (270, 270))
        return img_bin

    def load_training_set(self):
        data = []
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
                if (res > 1000 and label=="three"):
                    self.board[i][j] = 5
                if (res > 1200 and label=="one"):
                    self.board[i][j] = 0


    def get_board(self):
        return self.board
