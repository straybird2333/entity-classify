# python ray_ner_local.py --source zh_q4_news_10_100
# python ray_ner_local.py --source baijiahao
# python ner_label.py --source zh_q4_news_10_100
# python ner_label.py --source baijiahao
# python ray_ner_local.py --source qikan_chinese --rename text
# python ray_ner_local.py --source xuexiqiangguo --rename text

python ray_ner_local.py --source people_daily_new --rename text
python ner_label.py --source people_daily_new


