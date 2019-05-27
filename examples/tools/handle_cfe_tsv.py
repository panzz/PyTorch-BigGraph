import os
import json

examples_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
fileName = os.path.basename(os.path.realpath(__file__))
# print(examples_folder, fileName)


def main():
  graph_vector_file = examples_folder + '/data/cfe/cfe_joined_embeddings.tsv'
  graph_targets = examples_folder + '/data/cfe/cfe-targes.txt'

  datafile = examples_folder + '/data/cfe/cfe_data.tsv'
  labelfile = examples_folder + '/data/cfe/cfe_label.tsv'
  nheader = '/m/'
  dictionaryFile = examples_folder + '/data/cfe/dictionary.json'
  try:
    # with open(datafile, "w") as f2:
    #     f2.write('')
    # reset output file
    if os.path.exists(datafile):
      with open(datafile, 'r+', encoding='utf-8') as f:
          res = f.readlines()
          print(res)
          f.seek(0)
          f.truncate()
    if os.path.exists(labelfile):
      with open(labelfile, 'r+', encoding='utf-8') as f:
          res = f.readlines()
          print(res)
          f.seek(0)
          f.truncate()
    # format lables property
    targets = []
    if os.path.exists(graph_targets):
      with open(graph_targets, 'r+', encoding='utf-8') as f:
          lines = f.readlines()
          for i, l in enumerate(lines):
            targets.append(l.split('\n')[0])
          print(targets)

    # get max entities lines
    maxlineLen = 0
    with open(dictionaryFile, "r", encoding='utf-8') as f:
        json_obj = json.loads(f.read())
        maxlineLen = len(json_obj['entities']['all'])
        print(f'maxlineLen: {maxlineLen}')
   
    # open input file
    with open(graph_vector_file, "r") as f:
      lines = f.readlines()
      print(len(lines), type(lines))
      for i, l in enumerate(lines):
        nll = l.split('\t')
        if i < maxlineLen:
            with open(labelfile, 'a') as f2:
              writestr = nll[0] + '\t'+('0' if nll[0] in targets else '1')+'\n'
              f2.write(writestr)
            del nll[0]
            nllstr = '\t'.join(nll)
            with open(datafile, 'a') as f3:
              f3.write(nllstr)
        else:
          print('ignore i/total:{}/{}'.format(i, len(lines)))
          # with open(datafile, "a") as f1:
          #   f1.write(nllstr)
      print(f'write {labelfile} and {datafile} successful!')
  except:
      print("打开文件异常")


if __name__ == "__main__":
    main()
