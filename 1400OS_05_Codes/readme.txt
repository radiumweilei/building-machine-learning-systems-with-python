stackoverflow.com-Posts.7z 已经存储在移动硬盘 /code/python/data/building-machine-learning-systems-with-python/1400OS_05_Codes/ 目录下, 12G
用keka解压后得到58G的Posts.xml


使用步骤如下:
(
    1. 解压 stackoverflow.com-Posts.7z (12G), 得到 Posts.xml (58G)
    2. 过滤出 2011-12 的数据:
        cat Posts.xml | egrep -e "<\?xml version=\"1.0\" encoding=\"utf-8\"\?>|<posts>|</posts>|CreationDate=\"2011-12" > 2011-12.xml
)
注: 1,2 两步的结果文件 2011-12.xml 已经压缩成 2011-12.xml.tgz 并上传到git中, 直接从git中获取并解压即可往下运行本章所有代码
3. 将 2011-12.xml 转换成 filtered.tsv 和  filtered-meta.json
    so_xml_to_tsv.py
4. 移除异常字符: (这里的"^M"要使用"CTRL-V CTRL-M"生成，而不是直接键入"^M")
    sed -ie 's/^M/\n/g' filtered.tsv
    rm -rf filtered.tsve
5. 将 filtered.tsv 和 filtered-meta.json 转换成 chosen.tsv 和 chosen-meta.json
    chose_instances.py
6. 下载 punkt
  python
  >>> import nltk
  >>> nltk.download('punkt')
