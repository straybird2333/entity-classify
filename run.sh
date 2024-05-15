DATASET_LIST="stack_overflow"
# python batch_ray_ner_local.py --source ${DATASET_LIST}
# python batch_ray_classify_local.py --source ${DATASET_LIST} --stage tags
python batch_ner_label.py --source ${DATASET_LIST} --stage tag-domain
#zh.novel.ijjjxsw_2023_11_14__2023_12_01_17 dm_math_clean

# python filter.py --source feilu --word a
# python filter.py --source dm_math_clean --word a
# python filter.py --source gov_safety --word a
# mnbvc_law   stack_overflow