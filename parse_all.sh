#!/bin/bash

FILES=`find data -name *.xml`
OUT_DIR=output
mkdir -p ${OUT_DIR}
for p in $FILES; do
  f=${p##*/}
  output=${f%.xml}.txt
  echo "Processing $f..."
  python3 divide_xml_to_morphene.py -p $p > ${OUT_DIR}/${output}
done
