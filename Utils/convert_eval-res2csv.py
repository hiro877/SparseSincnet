import argparse
import csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 1.インスタンスの作成
    parser.add_argument('-i')  # 2.必要なオプションを追加
    args = parser.parse_args()  # 3.オプションを解析

    dict = {}
    tmp_list =[]
    count = 0
    with open(args.i) as f:
        lists = f.readlines()
        # print(lists)
        for l in lists:
            # print(l)
            if count % 2 == 0:
                # print(l.split(" ")[-1].split("=")[-1])
                tmp_list.append(l.split('/')[-1].split("_")[-1].split(".")[0])
            else:
                # print(l.split('/')[-1].split("_")[-1].split(".")[0])
                tmp_list.append(l.split(" ")[-1].split("=")[-1].strip())
                dict[int(tmp_list[0])] = float(tmp_list[1])
                print(tmp_list)
                tmp_list = []
            count += 1

    result = sorted(dict.items(), key=lambda i: i[0])
    print(result)


    with open(args.i.split(".")[0]+'.csv', 'w', newline='') as student_file:
        writer = csv.writer(student_file)
        writer.writerow(["Epoch", "err_te_snt"])
        for k, v in result:
            print(k, v)
            writer.writerow([k, v])