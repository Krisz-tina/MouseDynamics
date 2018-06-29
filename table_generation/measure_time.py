import csv
from utils import settings

def main(file_name):
    with open(file_name, 'r') as csv_file:
        data_reader = csv.reader(csv_file, delimiter=',')
        user_ids = ['7', '9', '12', '15', '16', '20', '21', '23', '29', '35']
        row = next(data_reader)
        row = next(data_reader)
        data = []
        k = 10
        for i in range(0, len(user_ids)):
            user_id = user_ids[i]
            counter = 0
            elapsed_time = 0
            counter_k = 0
            avg = 0
            avgavg = 0
            while row[-1] == user_id:

                if counter_k < k:
                    avg += float(row[0])
                    # print(row[0])
                else:
                    counter_k = 0
                    # avg /= k
                    # print('avg ' + str(avg))
                    avgavg += avg
                    avg = 0
                    counter += 1
                counter_k += 1

                # counter += 1
                # elapsed_time += float(row[0])
                try:
                    row = next(data_reader)
                except StopIteration:
                    data_row = [user_id, counter,  elapsed_time / counter]
                    data.append(data_row)
                    print('USER' + str(user_id))
                    # print(counter)
                    # print(avgavg)
                    print(avgavg / counter)
                    return data
            print('USER' + str(user_id))
            # print(counter)
            # print(avgavg)
            print(avgavg / counter)
            data_row = [user_id, counter, elapsed_time / counter]
            data.append(data_row)

        return data

def main2(file_name):
    with open(file_name, 'r') as csv_file:
        data_reader = csv.reader(csv_file, delimiter=',')
        user_ids = ['7', '9', '12', '15', '16', '20', '21', '23', '29', '35']
        row = next(data_reader)
        row = next(data_reader)
        data = []
        for i in range(0, len(user_ids)):
            user_id = user_ids[i]
            sum = 0
            while row[-1] == user_id:
                sum += float(row[0])
                try:
                    row = next(data_reader)
                except StopIteration:
                    data_row = [user_id, sum]
                    data.append(data_row)
                    print('USER' + str(user_id))
                    print(sum)
                    return data
            print('USER' + str(user_id))
            print(sum)
            data_row = [user_id, sum]
            data.append(data_row)

        return data


main2('D:/Sapientia EMTE/final exam/softwares/MouseDynamics/output/Book1.csv')