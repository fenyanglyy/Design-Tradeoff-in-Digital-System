# Assignment1

## Requirements

1. 怎样找到一些CIF或者QCIF的视频？

   URL：https://media.xiph.org/video/derf/

1. 视频播放器推荐：[VLC](https://www.videolan.org/)

2. 如何得到YUV格式？

   使用ffempg进行操作，参考脚本：

   ```bash
   ffmpeg -i input.y4m -c:v rawvideo -pix_fmt yuv420p output.yuv
   ```

​	BTW ffmpeg也可以直接把视频转化为任意想要的格式（例如从420p转化为444p，或者将视频转化为png图片序列，或者将图片序列转化为一个视频。）这个可以作为一种已实现的库作为验证。

