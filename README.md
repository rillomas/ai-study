Data is taken from http://elaws.e-gov.go.jp/download/lawdownload.html.
You need to put [system_core.dic](https://github.com/WorksApplications/Sudachi/releases/download/v0.1.0/sudachi-0.1.0-dictionary-core.zip)
inside the bin folder in order to make the scripts work.
```
$ parse_all.sh
$ cat output/* > merged.txt
$ python3 analyze.py -i merged.txt
```

