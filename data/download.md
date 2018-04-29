# Please Download vehicles and non-vehicles dataset

[vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) 和 [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) 的 64x64 影像都是由 [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html) 和 [KITTI vision benchmark suite ](http://www.cvlibs.net/datasets/kitti/) 這兩個dataset影片中擷取出來的。

```shell
wget https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip
wget https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip

unzip vehicles.zip
rm -r __MACOSX
unzip non-vehicles.zip
rm -r __MACOSX
rm vehicles.zip non-vehicles.zip
```

如果要增加訓練資料可以由Udacity最新的資料集中下載更多影像。
Link: [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) 



# Udacity Dataset

### Dataset 1

```
wget http://bit.ly/udacity-annoations-crowdai
mv udacity-annoations-crowdai object-detection-crowdai.tar.gz
tar xvf object-detection-crowdai.tar.gz
```

### Dataset 2

```
wget http://bit.ly/udacity-annotations-autti
mv udacity-annotations-autti object-dataset.tar.gz
tar zxvf object-dataset.tar.gz
```






# COCO Dataset

1. Install COCO Dataset

   ```shell
   cd <dataset_path>
   mkdir coco
   cd coco
   wget http://images.cocodataset.org/zips/train2014.zip
   wget http://images.cocodataset.org/zips/val2014.zip
   wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
   unzip http://images.cocodataset.org/zips/train2014.zip
   unzip http://images.cocodataset.org/zips/val2014.zip
   unzip http://images.cocodataset.org/annotations/annotations_trainval2014.zip
   ```

   ​

2. pip Install

   ```shell
   pip3 install baker
   pip3 install -I path.py==7.7.1
   pip3 install cytoolz
   pip3 install lxml
   ```

   ​

3. [**coco2pascal.py**](https://gist.github.com/chicham/6ed3842d0d2014987186#file-coco2pascal-py)

   ```python
   import baker
   import json
   from path import path
   from cytoolz import merge, join, groupby
   from cytoolz.compatibility import iteritems
   from cytoolz.curried import update_in
   from itertools import starmap
   from collections import deque
   from lxml import etree, objectify
   from scipy.io import savemat
   from scipy.ndimage import imread

   def keyjoin(leftkey, leftseq, rightkey, rightseq):
       return starmap(merge, join(leftkey, leftseq, rightkey, rightseq))

   def root(folder, filename, width, height):
       E = objectify.ElementMaker(annotate=False)
       return E.annotation(
               E.folder(folder),
               E.filename(filename),
               E.source(
                   E.database('MS COCO 2014'),
                   E.annotation('MS COCO 2014'),
                   E.image('Flickr'),
                   ),
               E.size(
                   E.width(width),
                   E.height(height),
                   E.depth(3),
                   ),
               E.segmented(0)
               )

   def instance_to_xml(anno):
       E = objectify.ElementMaker(annotate=False)
       xmin, ymin, width, height = anno['bbox']
       return E.object(
               E.name(anno['category_id']),
               E.bndbox(
                   E.xmin(xmin),
                   E.ymin(ymin),
                   E.xmax(xmin+width),
                   E.ymax(ymin+height),
                   ),
               )

   @baker.command
   def write_categories(coco_annotation, dst):
       content = json.loads(path(coco_annotation).expand().text())
       categories = tuple( d['name'] for d in content['categories'])
       savemat(path(dst).expand(), {'categories': categories})

   def get_instances(coco_annotation):
       coco_annotation = path(coco_annotation).expand()
       content = json.loads(coco_annotation.text())
       categories = {d['id']: d['name'] for d in content['categories']}
       return categories, tuple(keyjoin('id', content['images'], 'image_id', content['annotations']))

   def rename(name, year=2014):
           out_name = path(name).stripext()
           # out_name = out_name.split('_')[-1]
           # out_name = '{}_{}'.format(year, out_name)
           return out_name

   @baker.command
   def create_imageset(annotations, dst):
       annotations = path(annotations).expand()
       dst = path(dst).expand()
       val_txt = dst / 'val.txt'
       train_txt = dst / 'train.txt'

       for val in annotations.listdir('*val*'):
           val_txt.write_text('{}\n'.format(val.basename().stripext()), append=True)

       for train in annotations.listdir('*train*'):
           train_txt.write_text('{}\n'.format(train.basename().stripext()), append=True)

   @baker.command
   def create_annotations(dbpath, subset, dst):
       annotations_path = path(dbpath).expand() / 'annotations/instances_{}2014.json'.format(subset)
       images_path = path(dbpath).expand() / 'images/{}2014'.format(subset)
       categories , instances= get_instances(annotations_path)
       dst = path(dst).expand()

       for i, instance in enumerate(instances):
           instances[i]['category_id'] = categories[instance['category_id']]

       for name, group in iteritems(groupby('file_name', instances)):
           img = imread(images_path / name)
           if img.ndim == 3:
               out_name = rename(name)
               annotation = root('VOC2014', '{}.jpg'.format(out_name), 
                                 group[0]['height'], group[0]['width'])
               for instance in group:
                   annotation.append(instance_to_xml(instance))
               etree.ElementTree(annotation).write(dst / '{}.xml'.format(out_name))
               print out_name
           else:
               print instance['file_name']

   if __name__ == '__main__':
       baker.run()
   ```

   ​

4. Command Line

   ```shell
   mkdir voc
   mkdir voc/train
   mkdir cov/val
   python3 coco2voc.py create_annotations /coco train /voc/train
   python3 coco2voc.py create_annotations /coco val /voc/val
   ```

   ​

