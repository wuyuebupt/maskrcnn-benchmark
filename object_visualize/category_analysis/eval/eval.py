# %matplotlib inline
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np


annType = ['segm','bbox','keypoints']
annType = annType[1]      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
print ('Running demo for *%s* results.'%(annType))



#initialize COCO ground truth api
dataDir='/local3/dataset/coco/'
# dataType='val2014'
dataType='minival2014'
annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)
cocoGt=COCO(annFile)

#initialize COCO detections api
resFile='/local3/maskrcnn/iccv19/maskrcnn-benchmark-neighbor-v15-double-head-visualization-occlusion/object_visualize/category_analysis/eval/bbox.json'
resFile = resFile
# resFile = resFile%(dataDir, prefix, dataType, annType)
cocoDt=cocoGt.loadRes(resFile)


imgIds=sorted(cocoGt.getImgIds())
## toy 100 results for test
# imgIds=imgIds[0:100]
# imgId = imgIds[np.random.randint(100)]



# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
