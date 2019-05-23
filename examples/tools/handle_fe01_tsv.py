def main():
  filename = '/Users/panzz/Vobs/_dev/ai/_torch/PyTorch-BigGraph/examples/data/fe01/fe01_data.txt'
  outfile = '/Users/panzz/Vobs/_dev/ai/_torch/PyTorch-BigGraph/examples/data/fe01/fe01_output.txt'
  labelfile = '/Users/panzz/Vobs/_dev/ai/_torch/PyTorch-BigGraph/examples/data/fe01/fe01_label.txt'
  nheader = '/m/'
  theader = '/frontend/'
  try:
    # with open(outfile, "w") as f2:
    #     f2.write('')
    with open(outfile, 'r+', encoding='utf-8') as f:
        res = f.readlines()
        print(res)
        f.seek(0)
        f.truncate()
    with open(labelfile, 'r+', encoding='utf-8') as f:
        res = f.readlines()
        print(res)
        f.seek(0)
        f.truncate()

    with open(filename, "r") as f:
        lines = f.readlines()
        print(len(lines), type(lines))
        for l in lines:
          nll = l.split('\t')
          print('nll[{}]:{}'.format(type(nll), len(nll)))
          if nll[0].find(nheader) >= 0 or nll[0].find(theader) >= 0:
            with open(labelfile, 'a') as f2:
              f2.write(nll[0]+'\n')
            del nll[0]
          else:
            print('lines[{}]:{}'.format(lines.index(l), l))
          nllstr = '\t'.join(nll)
          
          with open(outfile, 'a') as f3:
            f3.write(nllstr)

          # with open(outfile, "a") as f1:
          #   f1.write(nllstr)
  except:
      print("打开文件异常")


if __name__ == "__main__":
    main()
