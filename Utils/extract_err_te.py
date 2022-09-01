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
            tmp_list.append(l.split(" ")[-1].split("=")[-1].strip())

    print(tmp_list)


    with open(args.i.split(".")[0]+'.csv', 'w', newline='') as student_file:
        writer = csv.writer(student_file)
        writer.writerow(["noise", "err_te_snt"])
        noise = 0.001
        for err in tmp_list:
            print(err)
            writer.writerow([noise, err])
            noise+=0.001