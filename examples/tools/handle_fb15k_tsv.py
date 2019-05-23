import os

examples_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
fileName = os.path.basename(os.path.realpath(__file__))
# print(examples_folder, fileName)

def main():
  filename = examples_folder + '/data/fb15k/fb15k_joined_embeddings.tsv'
  datafile = examples_folder + '/data/fb15k/fb15k_data.tsv'
  labelfile = examples_folder + '/data/fb15k/fb15k_label.tsv'
  nheader = '/m/'
  try:
    # with open(datafile, "w") as f2:
    #     f2.write('')
    # reset output file
    with open(datafile, 'r+', encoding='utf-8') as f:
        res = f.readlines()
        print(res)
        f.seek(0)
        f.truncate()
    with open(labelfile, 'r+', encoding='utf-8') as f:
        res = f.readlines()
        print(res)
        f.seek(0)
        f.truncate()
    # open input file
    with open(filename, "r") as f:
        lines = f.readlines()
        print(len(lines), type(lines))
        for l in lines:
          nll = l.split('\t')
          # print('nll[{}]:{}'.format(type(nll), len(nll)))
          if nll[0].find(nheader) >= 0:
            with open(labelfile, 'a') as f2:
              f2.write(nll[0]+'\n')
            del nll[0]
            nllstr = '\t'.join(nll)
            with open(datafile, 'a') as f3:
              f3.write(nllstr)
          else:
            # print('lines[{}]:{}'.format(lines.index(l), l))
            pass
          

          # with open(datafile, "a") as f1:
          #   f1.write(nllstr)
  except:
      print("打开文件异常")


if __name__ == "__main__":
    main()
