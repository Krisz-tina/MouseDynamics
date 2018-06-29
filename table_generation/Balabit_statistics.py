import csv
from utils import settings


def get_features():
    with open(settings.BALABIT_FEATURES_INPUT, 'r') as csv_file:
        data_reader = csv.reader(csv_file, delimiter=',')
        user_ids = ['7', '9', '12', '15', '16', '20', '21', '23', '29', '35']
        row = next(data_reader)
        row = next(data_reader)
        data = []
        for i in range(0, len(user_ids)):
            user_id = user_ids[i]
            counter = 0
            traveled_distance = 0
            elapsed_time = 0
            length_of_line = 0
            largest_deviation = 0
            while row[-1] == user_id:
                counter += 1
                traveled_distance += float(row[1])
                elapsed_time += float(row[2])
                length_of_line += float(row[3])
                largest_deviation += float(row[4])
                try:
                    row = next(data_reader)
                except StopIteration:
                    data_row = [user_id, counter, traveled_distance / counter, elapsed_time / counter,
                                length_of_line / counter,
                                largest_deviation / counter]
                    data.append(data_row)
                    return data
            print(counter)
            print(elapsed_time)
            print('travel' + str(traveled_distance/counter))
            data_row = [user_id, counter, elapsed_time/60, traveled_distance/counter, length_of_line/counter,
                        largest_deviation/counter]
            data.append(data_row)
        return data


def create_table(data):
    with open(settings.BALABIT_FEATURES_OUTPUT, 'w', newline='') as csv_file:
        data_writer = csv.writer(csv_file, delimiter=',')
        header = ['Felhasználó', 'Műveletek száma', 'Átlagos eltelt idő', 'Átlagos megtett út',
                  'Átlagos vonalhossz', 'Átlagos legnagyobb eltérés']
        header = ['Felhasznalo', 'Muveletek szama', 'Eltelt ido', 'Atlagos megtett ut',
                  'Atlagos vonalhossz', 'Atlagos legnagyobb elteres']
        data_writer.writerow(header)
        for i in range(len(data)):
            row = data[i]
            data_writer.writerow(row)




def main():
    data = get_features()
    print(data)
    print(len(data))
    create_table(data)


main()