需要有预先标定的xmp文件在对应的image文件旁

1. process_all 调整 (0,1,1)为你需要的（起始帧，step，最终帧）
	有时需要设置
2. process_one 调整对应的路径（建议修改成变量形式，我当时用了绝对路径）
	BD.rcbox指的是重建区域的bbox文件
	对应连续帧的文件夹下需要创建models与projects文件夹