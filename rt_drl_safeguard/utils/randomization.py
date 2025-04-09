
import numpy as np
from numpy.random import exponential
from scipy.stats import expon


def exp_delay(x, y=None):
    return exponential(scale=x, size=y)

class DelayTimeDistribution:
    """
    Represents a delay time distribution defined by bins and probabilities.
    """

    def __init__(self, bins, probabilities=None):
        """

        Args:
            bins: A list or NumPy array representing the bin edges (partition).
            probabilities: A list or NumPy array representing the probability
                           distribution over the bins.

        Raises:
            ValueError: If bins are not strictly increasing, probabilities are not
                        non-negative with a positive sum, or bin and probability
                        sizes are incompatible.
        """
        bins = np.array(bins)
        if not self._is_strictly_increasing(bins):
            raise ValueError("Bins must be strictly increasing.")

        self.size = len(bins) - 1
        self.bins = bins

        if probabilities is not None:
            probabilities = np.array(probabilities)
            if not self._is_non_negative_and_positive_sum(probabilities):
                raise ValueError("Probabilities must be non-negative with a positive sum.")

            if len(bins) - len(probabilities) != 1:
                raise ValueError("Number of bin endpoints must be one more than the number of probabilities.")
        else:
            probabilities = np.full((self.size,), 1.0 / self.size)

        self.probabilities = probabilities / np.sum(probabilities)


    def sample(self, num_samples=1):
        """
        Samples values from the distribution.

        Args:
            num_samples: The number of samples to generate.

        Returns:
            A NumPy array of sampled values.
        """
        bin_indices = np.random.choice(self.size, size=num_samples, p=self.probabilities)
        low_bounds = self.bins[bin_indices]
        high_bounds = self.bins[bin_indices + 1]
        return np.random.uniform(low_bounds, high_bounds)

    @staticmethod
    def _is_strictly_increasing(data):
        """
        Checks if a list or NumPy array is strictly increasing.
        """
        if len(data) <= 1:
            return True
        return np.all(np.diff(data) > 0)

    @staticmethod
    def _is_non_negative_and_positive_sum(data):
        """
        Checks if a list or NumPy array is non-negative and has a positive sum.
        """
        return np.all(np.array(data) >= 0) and np.sum(data) > 0

    def generate_above(self, point):
        """
        Generates a new Distribution object with bins above the given point.

        Args:
            point: The point to generate bins above.

        Returns:
            A new Distribution object.

        Raises:
            ValueError: If no bins can be generated above the point.
        """
        if self.bins[-1] <= point:
            raise ValueError("No bins can be generated above the point.")

        if self.bins[0] >= point:
            return self

        indices = np.where(self.bins > point)[0]
        index = indices[0]
        new_bins = np.concatenate(([point], self.bins[index+1:]))
        p = self.probabilities[index-1]
        p_scaled = p * (self.bins[index] - point) / (self.bins[index] - self.bins[index-1])
        new_probabilities = np.concatenate(([p_scaled], self.probabilities[index:]))
        return DelayTimeDistribution(new_bins, new_probabilities)

    def generate_below(self, point):
        """
        Generates a new Distribution object with bins below the given point.

        Args:
            point: The point to generate bins below.

        Returns:
            A new Distribution object.

        Raises:
            ValueError: If no bins can be generated below the point.
        """
        if self.bins[0] >= point:
            raise ValueError("No bins can be generated below the point.")

        if self.bins[-1] <= point:
            return self

        indices = np.where(self.bins < point)[0]
        index = indices[-1]
        new_bins = np.concatenate((self.bins[:index+1], [point]))
        p = self.probabilities[index]
        p_scaled = p * (point - self.bins[index+1]) / (self.bins[index+1] - self.bins[index])
        new_probabilities = np.concatenate((self.probabilities[:index], [p_scaled]))
        return DelayTimeDistribution(new_bins, new_probabilities)

    def cumulative_probability_above(self, point):
        if self.bins[-1] <= point:
            return 0
        elif self.bins[0] >= point:
            return 1
        else:
            indices_above = np.where(self.bins > point)[0]
            index = indices_above[0]
            p = self.probabilities[index-1]
            p_new = p * (self.bins[index] - point) / (self.bins[index] - self.bins[index-1])
            for i in range(index[0], len(self.probabilities)):
                p_new += self.probabilities[i]
            return p_new

    def cumulative_probability_below(self, point):
        if self.bins[-1] >= point:
            return 0
        elif self.bins[0] <= point:
            return 1
        else:
            indices_below = np.where(self.bins < point)[0]
            index = indices_below[0]
            p = self.probabilities[index]
            p_new = p * (point - self.bins[index+1]) / (self.bins[index+1] - self.bins[index])
            for i in range(0, index):
                p_new += self.probabilities[i]
            return p_new

    def print(self):
        print("Delay time distribution bins: {}, probabilities: {}".format(self.bins, self.probabilities))

    @staticmethod
    def create_from_exp_dist(bins, scale):
        cum_probs = expon.cdf(bins, scale=scale)
        probabilities = []
        for i in range(len(bins)-1):
            probabilities.append(cum_probs[i+1] - cum_probs[i])
        return DelayTimeDistribution(bins, np.array(probabilities))


class SwitchConfigurationSampler:
    def __init__(self, bins, probabilities=None):

        self.size = len(bins)
        self.bins = bins

        if probabilities is not None:
            probabilities = np.array(probabilities)
            if not self._is_non_negative_and_positive_sum(probabilities):
                raise ValueError("Probabilities must be non-negative with a positive sum.")

            if len(bins) != len(probabilities):
                raise ValueError("Number of bins must be the same as the number of probabilities.")
        else:
            probabilities = np.full((self.size,), 1.0 / self.size)

        self.probabilities = probabilities / np.sum(probabilities)

    def sample(self):
        """
        Samples a bin from the discrete distribution.

        Returns:
            The sampled bin (either a label from the 'bins' list or an index if 'bins' was not provided).
        """
        idx = np.random.choice(self.size, p=self.probabilities)
        return self.bins[idx]

    def get(self, i=0):
        """

        Get the i-th sample deterministically
        """
        return self.bins[i]

    def print(self):
        print("Switch configuration sampler bins: {}, probabilities: {}".format(self.bins, self.probabilities))

    @staticmethod
    def _is_non_negative_and_positive_sum(data):
        """
        Checks if a list or NumPy array is non-negative and has a positive sum.
        """
        return np.all(np.array(data) >= 0) and np.sum(data) > 0
