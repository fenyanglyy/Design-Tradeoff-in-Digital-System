# single instance for global config
class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance.init_config()  # Initialize the configuration
        return cls._instance

    # Add new global config here
    def init_config(self) -> None:
        self.width = 352
        self.height = 288
        self.block_size = 16
        self.search_range = 4
        self.residual_n = 3
        self.fps = 30
        
        # True when doing ex3
        self.Y_only_mode = True
        self.I_Period = 8  # 设置I帧的周期，例如每隔10个帧一个I帧
        self.QP = 4
        self.Default_start_I_mode = 0 # 默认帧内预测的初始模式为0 (0:horizen, 1:vertical)
        self.Default_start_mv = [0, 0, 0]
        self.golomb_m = 8

        # Prediction Feature
        # Multiple Reference Frames
        self.nRefFrames = 3
        # Variable Block Size
        self.VBSEnable = True
        # Fractional Motion Estimation
        self.FMEEnable = False
        # Fast Motion Estimation
        self.FastME = False

        # Feature Parameter
        self.VBS_lambda_constant = 0.085
