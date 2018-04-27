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