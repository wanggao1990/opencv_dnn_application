# Caffe-enet

A modified version of Caffe which supports the ENet architecture [github repo website](https://github.com/TimoSaemann/ENet).
It's caffemodel and protxt file cant not applied in OpenCv directly. [download](https://github.com/TimoSaemann/ENet)

Upsamle layer and BN layer has been re-implentmented in opencv, and modefied prototxt file to adapt new layers version. More detail in [model](/model) folder.

**Requirement:**
opencv version 4.x

test results:
![ALT_TEXT](example_image/pred1.jpg)

![ALT_TEXT](example_image/pred2.jpg)