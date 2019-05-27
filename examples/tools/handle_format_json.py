#!/usr/bin/env python
#-*- coding:utf-8 -*-

# Copyright (c) Lenovo, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import os
examples_folder = os.path.dirname(os.path.realpath(__file__))
fileName = os.path.basename(os.path.realpath(__file__))

outputfile = "test.json"
nerFilePath = '/fixtures/cfe_mtr100_mte100-train.txt'

def load_ner_file(filepath):
  if os.path.exists(filepath):
      with open(filepath, 'r+', encoding='utf-8') as f:
          lines = f.readlines()
          for i, l in enumerate(lines):
            targets = l.split('\t')
            print(f"{i}: {targets[0]} > {targets[2]} ")
  else:
    print(f'{filepath} is not exist')

def save_json_file(filepath):
    color = '#4f19c8'
    x = -739.36383
    y = -404.26147
    source = 'jquery'
    target = 'jsdom'
    node = {
        "color": f"{color}",
        "label": f"{source}",
        "attributes": {},
        "y": y,
        "x": x,
        "id": f"{source}",
        "size": 1
    }
    edge = {
        "sourceID": f"{source}",
        "attributes": {},
        "targetID": f"{target}",
        "size": 1
    }
    a = {"nodes": [], "edges": []}
    a["nodes"].append(node)
    a["edges"].append(edge)
    with open(filepath, "w", encoding='utf-8') as f:
        # indent 超级好用，格式化保存字典，默认为None，小于0为零个空格
        f.write(json.dumps(a, indent=4))
        # json.dump(a,f,indent=4)   # 和上面的效果一样

def load_json_file(filepath):
    with open(filepath, "r", encoding='utf-8') as f:
        aa = json.loads(f.read())
        f.seek(0)
        bb = json.load(f)    # 与 json.loads(f.read())
    print(aa)
    print(bb)

def main():
  print('start')
  # save_json_file(outputfile)
  print('main load file')
  # load_json_file(outputfile)
  load_ner_file(examples_folder+nerFilePath)
  print('end')

if __name__ == "__main__":
  main()
