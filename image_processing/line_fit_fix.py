import numpy as np


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = np.array([])
        # polynomial coefficients for new fit
        self.new_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = []
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def fit_fix(self):
        if any(self.new_fit):
            if any(self.best_fit):
                self.diffs = abs(self.new_fit - self.best_fit)
            if (self.diffs[0] > 0.001) or (self.diffs[1] > 1.0) or (self.diffs[2] > 100.):
                self.detected = False
            else:
                self.detected = True
                self.px_count = np.count_nonzero(self.allx)
                self.current_fit.append(self.new_fit)
                if len(self.current_fit) > 5:
                    del self.current_fit[0]
                self.best_fit = np.average(self.current_fit, axis=0)
        else:
            self.detected = False