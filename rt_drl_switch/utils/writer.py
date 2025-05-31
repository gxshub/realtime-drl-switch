import csv

class CSVWriter:
    def __init__(self, dir, filename):
        self.filename = dir + "/" + filename + ".csv"
        self.fieldnames = None

    def write(self, data_dict):
        """
        Writes a dictionary to the CSV file.

        Args:
            data_dict (dict): The dictionary to write. Assumes the keys are consistent across all logged dictionaries.
        """
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)

            if self.fieldnames is None:
                self.fieldnames = list(data_dict.keys())
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writeheader()

            if set(data_dict.keys()) != set(self.fieldnames):
                raise ValueError("Dictionary keys do not match the expected format.")

            writer.writerow(data_dict)

