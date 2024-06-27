%电子2101周冠 数字信号处理大作业
classdef Image_Processing_GUI < matlab.apps.AppBase
    
    % 与应用程序组件对应的属性
    properties (Access = public)
        %基本组件
        figure           matlab.ui.Figure
        Label_Name      matlab.ui.control.Label %姓名
        Label_Number      matlab.ui.control.Label %学号
        globalVar_LEN
        globalVar_THETA
        globalVar_PSF
        globalVar_NCORR
        globalVar_ICORR
        globalVar_K
        globalGau_VAR
        globalGau_MEAN
        globalDENSITY
        global_SPEVAR
        global_idealp_Fre
        global_ideahp_Fre
        global_ideabp_lFre
        global_ideabp_hFre
        global_ideabs_Fre
        global_ideabs_w
        global_butterOrder
        global_butterlp_Fre
        global_butterhp_Fre
        global_butterbp_lFre
        global_butterbp_hFre
        global_butterbs_Fre
        global_butterbs_w
        global_gausslp_Fre
        global_gausshp_Fre
        global_gaussbp_lFre
        global_gaussbp_hFre
        global_gaussbs_Fre
        global_gaussbs_w
        global_wtThreshold
        global_homeOrder %阶数
        global_Fre %截止频率
        global_hGain %高频增益
        global_lGain %低频增益
        global_MeanFilter %均值滤波领域大小
        global_MedianFilter %中值滤波领域大小
        %总变差滤波
        global_lambda  % 正则化参数，控制去噪程度
        global_numIterations % 迭代次数
        global_deltaT % 时间步长
        
        global_nlmeansH %去噪强度
        global_patchSize %块大小
        global_searchWindowSize %搜索窗口大小
        Label_PSNR      matlab.ui.control.Label %峰值信噪比
        Label_MSE      matlab.ui.control.Label %均方误差
        Label_SNR      matlab.ui.control.Label %信噪比
        Label_SSIM      matlab.ui.control.Label %结构相似性指数
        Label_qualityINFO      matlab.ui.control.Label %参数说明
        Label_QUALITY
        uibuttongroup1   matlab.ui.container.ButtonGroup
        %文件操作
        reset            matlab.ui.control.Button
        save             matlab.ui.control.Button
        exit             matlab.ui.control.Button
        load             matlab.ui.control.Button
        uibuttongroup2   matlab.ui.container.ButtonGroup
        %原始图像
        orgimg               matlab.ui.control.UIAxes
        uibuttongroup3   matlab.ui.container.ButtonGroup
        %效果预览图像
        effimg               matlab.ui.control.UIAxes
        uibuttongroup8   matlab.ui.container.ButtonGroup
        %频域滤波
        p21               matlab.ui.control.Button %理想低通滤波
        Label_idealp_Fre          matlab.ui.control.Label % 低通截止频率
        EditField_idealp_Fre          matlab.ui.control.NumericEditField %低通截止频率
        p22               matlab.ui.control.Button %理想高通滤波
        Label_ideahp_Fre          matlab.ui.control.Label % 高通截止频率
        EditField_ideahp_Fre          matlab.ui.control.NumericEditField %高通截止频率
        p23               matlab.ui.control.Button %理想带通滤波
        Label_ideabp_lFre          matlab.ui.control.Label % 带通低频截止频率
        EditField_ideabp_lFre          matlab.ui.control.NumericEditField %带通低频截止频率
        Label_ideabp_hFre          matlab.ui.control.Label % 带通高频截止频率
        EditField_ideabp_hFre          matlab.ui.control.NumericEditField %带通高频截止频率
        p24               matlab.ui.control.Button %理想带阻滤波
        Label_ideabs_Fre          matlab.ui.control.Label % 带通低频截止频率
        EditField_ideabs_Fre          matlab.ui.control.NumericEditField %带通低频截止频率
        Label_ideabs_w          matlab.ui.control.Label % 带通高频截止频率
        EditField_ideabs_w          matlab.ui.control.NumericEditField %带通高频截止频率
        %巴特沃斯滤波器
        Label_butterOrder          matlab.ui.control.Label % 巴特沃斯滤波器阶数
        EditField_butterOrder          matlab.ui.control.NumericEditField %巴特沃斯滤波器阶数
        p31               matlab.ui.control.Button %巴特沃斯低通
        Label_butterlp_Fre          matlab.ui.control.Label % 低通截止频率
        EditField_butterlp_Fre          matlab.ui.control.NumericEditField %低通截止频率
        p32               matlab.ui.control.Button %巴特沃斯高通
        Label_butterhp_Fre          matlab.ui.control.Label % 高通截止频率
        EditField_butterhp_Fre          matlab.ui.control.NumericEditField %高通截止频率
        p33               matlab.ui.control.Button %巴特沃斯带通
        Label_butterbp_lFre          matlab.ui.control.Label % 带通低频截止频率
        EditField_butterbp_lFre          matlab.ui.control.NumericEditField %带通低频截止频率
        Label_butterbp_hFre          matlab.ui.control.Label % 带通高频截止频率
        EditField_butterbp_hFre          matlab.ui.control.NumericEditField %带通高频截止频率
        p34               matlab.ui.control.Button %巴特沃斯带通
        Label_butterbs_Fre          matlab.ui.control.Label % 带通低频截止频率
        EditField_butterbs_Fre          matlab.ui.control.NumericEditField %带通低频截止频率
        Label_butterbs_w          matlab.ui.control.Label % 带通高频截止频率
        EditField_butterbs_w          matlab.ui.control.NumericEditField %带通高频截止频率
        f21               matlab.ui.control.Button  %高斯低通滤波
        Label_gausslp_Fre          matlab.ui.control.Label % 低通截止频率
        EditField_gausslp_Fre          matlab.ui.control.NumericEditField %低通截止频率
        f22               matlab.ui.control.Button  %高斯高通滤波
        Label_gausshp_Fre          matlab.ui.control.Label % 高通截止频率
        EditField_gausshp_Fre          matlab.ui.control.NumericEditField %高通截止频率
        f23               matlab.ui.control.Button  %高斯带通滤波
        Label_gaussbp_lFre          matlab.ui.control.Label % 带通低频截止频率
        EditField_gaussbp_lFre          matlab.ui.control.NumericEditField %带通低频截止频率
        Label_gaussbp_hFre          matlab.ui.control.Label % 带通高频截止频率
        EditField_gaussbp_hFre          matlab.ui.control.NumericEditField %带通高频截止频率
        f24               matlab.ui.control.Button  %高斯带阻滤波
        Label_gaussbs_Fre          matlab.ui.control.Label % 带通截止频率
        EditField_gaussbs_Fre          matlab.ui.control.NumericEditField %带通截止频率
        Label_gaussbs_w          matlab.ui.control.Label % 阻带带宽
        EditField_gaussbs_w          matlab.ui.control.NumericEditField % 阻带带宽
        p4               matlab.ui.control.Button %小波去噪
        Label_wtThreshold          matlab.ui.control.Label % 小波去噪阈值
        EditField_wtThreshold          matlab.ui.control.NumericEditField %小波去噪阈值
        homefilter    matlab.ui.container.ButtonGroup %同态滤波
        p5               matlab.ui.control.Button %同态滤波
        Label_homeOrder          matlab.ui.control.Label % 同态滤波阶数
        EditField_homeOrder          matlab.ui.control.NumericEditField %同态滤波阶数
        Label_Fre          matlab.ui.control.Label % 同态滤波截止频率
        EditField_Fre          matlab.ui.control.NumericEditField %同态滤波截止频率
        Label_hGain          matlab.ui.control.Label %同态滤波高频增益
        EditField_hGain          matlab.ui.control.NumericEditField %同态滤波高频增益
        Label_lGain          matlab.ui.control.Label % 同态滤波低频增益
        EditField_lGain          matlab.ui.control.NumericEditField %同态滤波低频增益
        uibuttongroup9   matlab.ui.container.ButtonGroup
        %空间滤波器/去噪
        p1               matlab.ui.control.Button %维纳滤波
        Label_Avg          matlab.ui.control.Label %均值滤波
        Label_MeanFilter          matlab.ui.control.Label % 均值滤波领域大小
        EditField_MeanFilter          matlab.ui.control.NumericEditField % 均值滤波领域大小
        f1_1               matlab.ui.control.Button  %均值滤波 replicate
        f1_2               matlab.ui.control.Button  %均值滤波 symmetric
        f1_3               matlab.ui.control.Button  %均值滤波 circular
        f3               matlab.ui.control.Button  %中值滤波
        Label_MedianFilter          matlab.ui.control.Label % 中值滤波领域大小
        EditField_MedianFilter          matlab.ui.control.NumericEditField % 中值滤波领域大小
        f4               matlab.ui.control.Button  %非局部均值去噪
        Label_nlmeansH   matlab.ui.control.Label%去噪强度
        EditField_nlmeansH   matlab.ui.control.NumericEditField
        Label_patchSize   matlab.ui.control.Label %块大小
        EditField_patchSize   matlab.ui.control.NumericEditField
        Label_searchWindowSize   matlab.ui.control.Label %搜索窗口大小
        EditField_searchWindowSize   matlab.ui.control.NumericEditField
        f5               matlab.ui.control.Button  %总变差去噪
        Label_lambda   matlab.ui.control.Label%去噪强度
        EditField_lambda   matlab.ui.control.NumericEditField
        Label_numIterations   matlab.ui.control.Label %块大小
        EditField_numIterations   matlab.ui.control.NumericEditField
        Label_deltaT   matlab.ui.control.Label %搜索窗口大小
        EditField_deltaT   matlab.ui.control.NumericEditField
        Label_Wn          matlab.ui.control.Label %维纳滤波提示
        %添加噪波
        uibuttongroup10  matlab.ui.container.ButtonGroup
        n1               matlab.ui.control.Button %高斯噪声
        Label_MEAN          matlab.ui.control.Label %均值
        Label_VAR          matlab.ui.control.Label %方差
        EditField_MEAN       matlab.ui.control.NumericEditField %运动长度输入
        EditField_VAR        matlab.ui.control.NumericEditField %运动方向角输入
        n2               matlab.ui.control.Button %泊松噪波
        n3               matlab.ui.control.Button %椒盐噪声
        Label_DENSITY          matlab.ui.control.Label %噪声密度
        EditField_DENSITY       matlab.ui.control.NumericEditField %噪声密度
        n4               matlab.ui.control.Button %斑点噪声
        Label_SPEVAR          matlab.ui.control.Label %斑点噪声方差
        EditField_SPEVAR       matlab.ui.control.NumericEditField %噪声密度
        n5               matlab.ui.control.Button %运动噪波
        Label_LEN          matlab.ui.control.Label %运动长度
        Label_THETA          matlab.ui.control.Label %运动方向角
        EditField_LEN       matlab.ui.control.NumericEditField %运动长度输入
        EditField_THETA        matlab.ui.control.NumericEditField %运动方向角输入
        uibuttongroup11  matlab.ui.container.ButtonGroup
        %变换波形显示
        g1               matlab.ui.control.UIAxes %图像RGB颜色分解
        g2               matlab.ui.control.UIAxes %傅里叶变换
        g3               matlab.ui.control.UIAxes %DCT变换
        g4               matlab.ui.control.UIAxes  %小波变换1
        g5               matlab.ui.control.UIAxes  %小波变换2
        g6               matlab.ui.control.UIAxes  %小波变换3
        g7               matlab.ui.control.UIAxes  %小波变换4
        uibuttongroup12_1      matlab.ui.container.ButtonGroup %小波选择
        Label            matlab.ui.control.Label
        Button1          matlab.ui.control.Button
        Button2          matlab.ui.control.Button
        Button3          matlab.ui.control.Button
        Button4          matlab.ui.control.Button
        Button5          matlab.ui.control.Button
        Button6          matlab.ui.control.Button
        Button7          matlab.ui.control.Button
    end
    
    %根据对图片施加的效果更新变换图像
    methods (Access = private)
        
        function update(app, handles) %更新图表
            % 显示等待条
            h = waitbar(0, '更新图像中 等待...');
            steps = 30;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            
            mysize=size(handles.img);             % 获取处理后的图像大小
            if numel(mysize)>2
                % 如果图像大小大于二维，则执行 updateColr 函数进行处理
                updateColr(app, handles)
                updateColr_FT(app, handles)
                updateColr_DCT(app, handles)
            else
                % 否则执行 updateGray 函数进行处理
                updateGray(app, handles)
                updateGray_FT(app, handles)
                updateGary_DCT(app, handles) %更新彩色图像彩色直方图
            end
            updateDWT(app, handles)
            
            [quality_MSE,quality_SNR,quality_PSNR,quality_SSIM] = calculateQualityMetrics(app,handles.i,handles.img);
            app.Label_SSIM.Text = ['SSIM: ', num2str(round(quality_SSIM, 3)), ' dB'];
            app.Label_PSNR.Text = ['PSNR: ', num2str(round(quality_PSNR, 3)), ' dB'];
            app.Label_SNR.Text = ['SNR: ', num2str(round(quality_SNR, 3)), ' dB'];
            app.Label_MSE.Text = ['MSE: ', num2str(round(quality_MSE, 3)), ''];
            
            
        end
        
        function updateColr(app, handles) %更新彩色图像彩色直方图
            set(handles.g1, 'Visible', 'on');
            % 将图像数据重塑为一维数组
            ImageData1 = mat2gray( handles.img(:,:,1) );
            ImageData1 = reshape(ImageData1, [numel(ImageData1), 1]) * 255;
            ImageData2 = mat2gray( handles.img(:,:,2) );
            ImageData2 = reshape(ImageData2, [numel(ImageData2), 1]) * 255;
            ImageData3 = mat2gray( handles.img(:,:,3) );
            ImageData3 = reshape(ImageData3, [numel(ImageData3), 1]) * 255;
            % 计算直方图
            [H1, X1] = imhist(uint8(ImageData1), 256); % 指定横坐标范围为0~255
            [H2, X2] = imhist(uint8(ImageData2), 256); % 指定横坐标范围为0~255
            [H3, X3] = imhist(uint8(ImageData3), 256); % 指定横坐标范围为0~255
            % 在g1示波器中绘制彩色直方图
            axes(handles.g1);
            cla;
            hold on; % 保持绘图区域，以便绘制其他曲线
            plot(X1, H1, 'R'); % 绘制红色通道直方图，颜色为红色
            plot(X2, H2, 'G'); % 绘制绿色通道直方图，颜色为绿色
            plot(X3, H3, 'B'); % 绘制蓝色通道直方图，颜色为蓝色
            legend('R','G','B');
            title('图像颜色直方图');
            maxY = max([H1 H2 H3]);
            maxY = max(maxY);
            disp(maxY);
            axis([0 255 0 maxY-1]); % 设置坐标轴范围，x轴范围为0到256，y轴范围根据直方图的最大值确定
            hold off; % 结束绘图区域的保持状态
        end
        
        function updateGray(app, handles) %更新灰度直方图
            set(handles.g1, 'Visible', 'on');
            % 将滤波后的图像数据重塑为一维数组，并将像素值从[0, 1]映射到[0, 255]
            ImageData = mat2gray(handles.img);
            ImageData = reshape(ImageData, [numel(ImageData), 1]) * 255;
            % 计算直方图
            [H, X] = imhist(uint8(ImageData), 256); % 指定横坐标范围为0~255
            % 在 g1 示波器中绘制灰色直方图
            axes(handles.g1);
            cla;
            hold on;
            bar(X, H, 'k');
            legend('灰度');
            title('图像灰度直方图');
            axis([0 255 0 max(H)]);
            hold off;
        end
        
        function updateColr_FT(app, handles) %更新彩色图像彩色直方图
            set(handles.g2, 'Visible', 'on');
            % 计算傅里叶变换
            imgft = zeros(size(handles.img));
            for i = 1:3 % 遍历每个颜色通道
                channel = handles.img(:,:,i);
                channel_ft = fft2(channel); % 傅里叶变换
                imgft(:,:,i) = abs(fftshift(channel_ft)); % 将傅里叶变换的模进行象限转换并保存
            end
            % 将三个颜色通道的傅里叶变换结果合并或做其他处理
            final_imgft = sum(imgft, 3); % 简单地将三个通道的结果相加
            % 可视化傅里叶变换结果
            imgftam = log(final_imgft + 1); % 将傅里叶变换结果的幅值映射到小的正数
            axes(handles.g2);
            cla;
            imshow(imgftam, []); % 显示傅里叶变换结果图像，映射到[0,1]
            title('彩色图像的傅里叶谱');
            hold off;
        end
        
        function updateGray_FT(app, handles) %更新彩色图像彩色直方图
            set(handles.g2, 'Visible', 'on');
            % 计算灰度傅里叶变换
            grayImg = handles.img;
            % 计算傅里叶变换
            imgft = fft2(grayImg); % 对灰度图像执行傅里叶变换
            imgft = abs(fftshift(imgft)); % 将傅里叶变换的模进行象限转换并保存
            % 可视化傅里叶变换结果
            imgftam = log(imgft + 1); % 将傅里叶变换结果的幅值映射到小的正数
            axes(handles.g2);
            cla;
            imshow(imgftam, []); % 显示傅里叶变换结果图像，映射到[0,1]
            title('灰度图像的傅里叶谱');
            hold off;
        end
        
        function updateColr_DCT(app, handles) %更新彩色图像彩色直方图
            set(handles.g3, 'Visible', 'on');
            % 对彩色图像进行DCT变换
            dct_img = zeros(size(handles.img));
            for i = 1:3 % 遍历每个颜色通道
                channel = handles.img(:,:,i);
                dct_channel = dct2(channel);
                dct_img(:,:,i) = dct_channel;
            end
            dct_img = log(abs(dct_img));
            axes(handles.g3);
            cla;
            imshow(dct_img, []);
            title('彩色图像的DCT频谱');
            hold off;
        end
        
        function updateGary_DCT(app, handles) %更新彩色图像彩色直方图
            set(handles.g3, 'Visible', 'on');
            % 对灰色图像进行DCT变换
            grayImg = handles.img;
            % 计算DCT变换
            dctImg = dct2(grayImg);
            % 可视化DCT变换结果
            axes(handles.g3);
            cla;
            imshow(log(abs(dctImg) + 1), []); % 显示DCT变换结果的幅值，映射到[0,1]
            title('灰度图像的DCT变换');
            hold off;
        end
        
        function updateDWT(app, handles) %更新小波变换
            set(handles.g4, 'Visible', 'on');
            set(handles.g5, 'Visible', 'on');
            set(handles.g6, 'Visible', 'on');
            set(handles.g7, 'Visible', 'on');
            
            % 计算小波变换
            mysize=size(handles.img);             % 获取处理后的图像大小
            if numel(mysize)>2
                grayimg = rgb2gray(handles.img);
            else
                grayimg = handles.img;
            end
            [grap_LLY, HL, LH, HH] = dwt2(grayimg, 'haar');
            [LLY, HL, LH, HH] = dwt2(handles.img, 'haar');
            %显示小波变换
            set(handles.g4, 'Visible', 'on');
            axes(handles.g4);
            cla;
            imshow(grap_LLY, []);
            hold off;
            title('小波变换(1.灰度低频近似值)');
            set(handles.g5, 'Visible', 'on');
            axes(handles.g5);
            cla;
            imshow(HL, []);
            title('2.水平方向细节');
            hold off;
            set(handles.g6, 'Visible', 'on');
            axes(handles.g6);
            cla;
            imshow(LH, []);
            title('3.垂直方向细节');
            hold off;
            set(handles.g7, 'Visible', 'on');
            axes(handles.g7);
            cla;
            imshow(HH, []);
            title('4.对角线方向细节');
            hold off;
        end
        
        % 定义计算图像质量评价指标的函数
        function [quality_MSE,quality_SNR,quality_PSNR,quality_SSIM] = calculateQualityMetrics(app,originalImage, processedImage, metric)
            % 将图像转为双精度类型
            originalImage = double(originalImage);
            processedImage = double(processedImage);
            
            quality_PSNR = calculatePSNR(app,originalImage, processedImage);
            quality_SSIM = calculateSSIM(app,originalImage, processedImage);
            quality_MSE = calculateMSE(app,originalImage, processedImage);
            quality_SNR = calculateSNR(app,originalImage, processedImage);
        end
        
        % 定义计算PSNR的函数
        function quality_PSNR = calculatePSNR(app,originalImage, processedImage)
            % 获取图像大小
            [M, N, ~] = size(originalImage);
            
            % 计算均方误差（MSE）
            mse = sum(sum((originalImage - processedImage).^2)) / (M * N);
            
            % 计算PSNR
            A = 255; % 8比特精度图像的最大像素值
            quality_PSNR = 10 * log10(A^2 / mse);
        end
        
        % 定义计算SSIM的函数
        function quality_SSIM = calculateSSIM(app,originalImage, processedImage)
            % 使用 MATLAB 内置的 ssim 函数计算 SSIM
            quality_SSIM = ssim(processedImage,originalImage);
        end
        
        % 定义计算MSE的函数
        function quality_MSE = calculateMSE(app,originalImage, processedImage)
            % 获取图像大小
            [M, N, ~] = size(originalImage);
            
            % 计算均方误差（MSE）
            quality_MSE = sum(sum((originalImage - processedImage).^2)) / (M * N);
        end
        
        % 定义计算 SNR 的函数
        function quality_SNR = calculateSNR(app,originalImage, processedImage)
            % 计算信号强度
            signal = sum(sum(originalImage.^2));
            
            % 计算噪声强度
            noise = sum(sum((originalImage - processedImage).^2));
            
            % 计算 SNR
            quality_SNR = 10 * log10(signal / noise);
        end
        
    end
    
    % 句柄控件控制的回调函数
    methods (Access = private)
        
        % 组件创建后执行的代码
        function Image_processing_GUI_OpeningFcn(app, varargin)
            % Create GUIDE-style callback args - Added by Migration Tool
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app); %#ok<ASGLU>
            
            % 选择 Image_processing_GUI 的默认命令行输出
            handles.output = hObject;
            % 禁用一些按钮和控件
            set(handles.load,'Enable','on');
            set(handles.save,'Enable','off');
            set(handles.exit,'Enable','off');
            set(handles.reset,'Enable','off');
            set(handles.effimg,'Visible','off');
            set(handles.orgimg,'Visible','off');
            set(handles.g1,'Visible','off');
            set(handles.g2,'Visible','off');
            set(handles.g3,'Visible','off');
            set(handles.g4,'Visible','off');
            set(handles.g5,'Visible','off');
            set(handles.g6,'Visible','off');
            set(handles.g7,'Visible','off');
            set(handles.n1,'Enable','off');
            set(handles.n2,'Enable','off');
            set(handles.n3,'Enable','off');
            set(handles.n4,'Enable','off');
            set(handles.f1_1,'Enable','off');
            set(handles.f1_2,'Enable','off');
            set(handles.f1_3,'Enable','off');
            set(handles.f21,'Enable','off');
            set(handles.f22,'Enable','off');
            set(handles.f23,'Enable','off');
            set(handles.f24,'Enable','off');
            set(handles.f3,'Enable','off');
            set(handles.f4,'Enable','off');
            set(handles.f5,'Enable','off');
            set(handles.p1,'Enable','off');
            set(handles.p21,'Enable','off');
            set(handles.p22,'Enable','off');
            set(handles.p23,'Enable','off');
            set(handles.p24,'Enable','off');
            set(handles.p31,'Enable','off');
            set(handles.p32,'Enable','off');
            set(handles.p33,'Enable','off');
            set(handles.p34,'Enable','off');
            set(handles.p4,'Enable','off');
            set(handles.p5,'Enable','off');
            set(handles.uibuttongroup12_1,'visible','off');
            set(handles.Button1,'Enable','off');
            set(handles.Button2,'Enable','off');
            set(handles.Button3,'Enable','off');
            set(handles.Button4,'Enable','off');
            set(handles.Button5,'Enable','off');
            set(handles.Button6,'Enable','off');
            set(handles.Button7,'Enable','off');
            set(handles.n5,'Enable','off');
            % 更新 handles 结构
            guidata(hObject, handles);
        end
        
        % 加载
        function load_Callback(app, event)
            % Create GUIDE-style callback args - Added by Migration Tool
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>
            
            % 打开文件选择对话框
            [file, path] = uigetfile({'*.jpg;*.bmp;*.jpeg;*.png', '图片文件 (*.jpg,*.bmp,*.jpeg,*.png)'}, '打开文件');
            
            image = [path file];
            handles.file = image;
            
            % 如果用户取消选择文件，弹出警告对话框
            if (file == 0)
                warndlg('您没有选择图片。') ;
                return;
            end
            
            % 获取文件扩展名
            [fpath, fname, fext] = fileparts(file);
            validex = {'.bmp', '.jpg', '.jpeg', '.png'};
            found = 0;
            
            % 检查文件扩展名是否合法
            for x = 1:length(validex)
                if (strcmpi(fext, validex{x}))
                    found = 1;
                    
                    % 启用一些按钮和控件
                    set(handles.save, 'Enable', 'on');
                    set(handles.exit, 'Enable', 'on');
                    set(handles.reset, 'Enable', 'on');
                    set(handles.effimg, 'Visible', 'on');
                    set(handles.orgimg, 'Visible', 'on');
                    set(handles.n1, 'Enable', 'on');
                    set(handles.n2, 'Enable', 'on');
                    set(handles.n3, 'Enable', 'on');
                    set(handles.n4, 'Enable', 'on');
                    set(handles.f1_1,'Enable','on');
                    set(handles.f1_2,'Enable','on');
                    set(handles.f1_3,'Enable','on');
                    set(handles.f21, 'Enable', 'on');
                    set(handles.f22, 'Enable', 'on');
                    set(handles.f23, 'Enable', 'on');
                    set(handles.f24, 'Enable', 'on');
                    set(handles.f3, 'Enable', 'on');
                    set(handles.f4, 'Enable', 'on');
                    set(handles.f5, 'Enable', 'on');
                    set(handles.p21,'Enable','on');
                    set(handles.p22,'Enable','on');
                    set(handles.p23,'Enable','on');
                    set(handles.p24,'Enable','on');
                    set(handles.p31,'Enable','on');
                    set(handles.p32,'Enable','on');
                    set(handles.p33,'Enable','on');
                    set(handles.p34,'Enable','on');
                    set(handles.p4, 'Enable', 'on');
                    set(handles.p5, 'Enable', 'on');
                    set(handles.Button1,'Enable','on');
                    set(handles.Button2,'Enable','on');
                    set(handles.Button3,'Enable','on');
                    set(handles.Button4,'Enable','on');
                    set(handles.Button5,'Enable','on');
                    set(handles.Button6,'Enable','on');
                    set(handles.Button7,'Enable','on');
                    set(handles.n5,'Enable','on');
                    
                    % 读取图像数据
                    handles.img = imread(image); %handles.img载入图像数据
                    handles.i = imread(image); %原图数据
                    %                     assignin('base', 'myVariable2', handles.img);
                    
                    % 显示图像在effimg和orgimg中
                    axes(handles.orgimg); % 设置orgimg为当前绘图区域
                    cla; % 清空当前绘图区域
                    imshow(handles.img); % 在orgimg绘图区域显示图像数据
                    
                    axes(handles.effimg); % 设置effimg为当前绘图区域
                    cla; % 清空当前绘图区域
                    imshow(handles.img); % 在effimg绘图区域显示图像数据
                    
                    
                    % 更新 handles 结构
                    guidata(hObject, handles);
                    
                    % 关闭等待条
                    hWaitbar = waitbar(0, '等待......', 'CreateCancelBtn', 'delete(gcbf)');
                    set(hWaitbar, 'Color', [0.9, 0.9, 0.9]);
                    steps = 3;
                    waitbar(1 / steps);
                    waitbar(2 / steps);
                    waitbar(3 / steps);
                    %关闭进度条
                    close(hWaitbar);
                    
                    update(app, handles);
                    
                    hWaitbar = waitbar(0, '等待......', 'CreateCancelBtn', 'delete(gcbf)');
                    set(hWaitbar, 'Color', [0.9, 0.9, 0.9]);
                    steps = 10;
                    for waiting = 1 : 10
                        waitbar(waiting / steps);
                    end
                    close(hWaitbar);
                    
                    set(handles.uibuttongroup12_1,'visible','on');
                    
                    app.globalVar_LEN = 10;
                    app.globalVar_THETA = 20;
                    app.globalGau_MEAN = 0;
                    app.globalGau_VAR = 0.05;
                    app.globalDENSITY = 0.1;
                    app.global_SPEVAR = 0.04;
                    %理想滤波器
                    app.global_idealp_Fre = 60;
                    app.global_ideahp_Fre = 30;
                    app.global_ideabp_lFre = 20;
                    app.global_ideabp_hFre = 80;
                    app.global_ideabs_Fre = 50;
                    app.global_ideabs_w = 30;
                    %巴特沃斯滤波器
                    app.global_butterOrder = 6;
                    app.global_butterlp_Fre = 60;
                    app.global_butterhp_Fre = 30;
                    app.global_butterbp_lFre = 20;
                    app.global_butterbp_hFre = 80;
                    app.global_butterbs_Fre = 50;
                    app.global_butterbs_w = 30;
                    %高斯滤波器
                    app.global_gausslp_Fre = 60;
                    app.global_gausshp_Fre = 30;
                    app.global_gaussbp_lFre = 20;
                    app.global_gaussbp_hFre = 80;
                    app.global_gaussbs_Fre = 50;
                    app.global_gaussbs_w = 30;
                    %小波去噪
                    app.global_wtThreshold = 0.5;
                    %同态滤波
                    app.global_homeOrder = 1;
                    app.global_Fre = 5;
                    app.global_hGain = 1.1;
                    app.global_lGain = 0.1;
                    %均值滤波领域大小
                    app.global_MeanFilter = 6;
                    %中值滤波领域大小
                    app.global_MedianFilter = 3;
                    %总变差滤波
                    app.global_lambda = 0.1;  % 正则化参数，控制去噪程度
                    app.global_numIterations = 30; % 迭代次数
                    app.global_deltaT = 0.1; % 时间步长
                    app.global_nlmeansH = 2; %去噪强度
                    app.global_patchSize = 5; %块大小
                    app.global_searchWindowSize = 10; %搜索窗口大小
                    
                    % 更新 handles 结构
                    guidata(hObject, handles);
                    break;
                end
            end
            % 如果文件扩展名不合法，弹出错误对话框
            if (found == 0)
                errordlg('文件扩展名不正确，请从可用扩展名[.jpg、.jpeg、.bmp、.png]中选择文件','Image Format Error');
            end
        end
        
        % 退出
        function exit_Callback(app, event)
            % Create GUIDE-style callback args - Added by Migration Tool
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>
            
            % 关闭所有图形窗口
            close all;
        end
        
        % 恢复原图
        function reset_Callback(app, event)
            % Create GUIDE-style callback args - Added by Migration Tool
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>
            % 将图像恢复为初始状态
            handles.img = handles.i;
            % 在effimg中显示图像
            axes(handles.effimg);
            cla;
            imshow(handles.img);
            % 更新
            update(app, handles);
            % 更新 handles 结构
            guidata(hObject,handles);
        end
        
        % 保存图像
        function save_Callback(app, event)
            % Create GUIDE-style callback args - Added by Migration Tool
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>
            
            % 弹出文件保存对话框
            [file, path] = uiputfile('*.jpg', 'Save Image as');
            
            % 拼接保存路径和文件名
            save = [path file];
            
            try
                % 将图像保存为 JPG 格式
                imwrite(handles.img, save, 'jpg');
            catch
                warndlg('您没有填写保存图片名称。') ;
                return;
            end
            
        end
        
        % f1_1 : 均值滤波  replicate
        function f1_1_Callback(app, event)
            % 定义一个名为 f1_1_Callback 的函数，其输入参数为 app、event 和 neighborhood_size
            % 创建 GUIDE 风格的回调函数参数（由迁移工具添加）
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>
            % 显示等待条
            h = waitbar(0, '等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            
            mean_size = [app.global_MeanFilter, app.global_MeanFilter];
            % 使用 convertToGUIDECallbackArguments 函数将 app 和 event 转换为 GUIDE 回调函数所需的参数，并将其存储在 hObject、eventdata 和 handles 变量中
            h = fspecial('average', mean_size);
            % 创建一个指定大小的平均滤波器模板
            handles.img = imfilter(handles.img, h, 'replicate');
            % 使用平均滤波器对图像进行滤波处理
            axes(handles.effimg);
            cla;
            imshow(handles.img)
            % 将当前坐标轴更改为 handles.effimg 所代表的坐标轴，清空坐标轴并显示处理后的图像
            % 更新 handles 结构
            guidata(hObject, handles);
            % 显示等待条
            h = waitbar(0, '更新图像中 等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            % 更新函数
            update(app, handles);
        end
        % f1_2 : 均值滤波 symmetric
        function f1_2_Callback(app, event)
            % 创建 GUIDE 风格的回调函数参数（由迁移工具添加）
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>
            % 显示等待条
            h = waitbar(0, '等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            mean_size = [app.global_MeanFilter, app.global_MeanFilter];
            % 使用 convertToGUIDECallbackArguments 函数将 app 和 event 转换为 GUIDE 回调函数所需的参数，并将其存储在 hObject、eventdata 和 handles 变量中
            h = fspecial('average', mean_size);
            % 创建一个平均滤波器模板
            handles.img=imfilter(handles.img,h,'symmetric');
            % 使用平均滤波器对图像进行滤波处理
            axes(handles.effimg);
            cla;
            imshow(handles.img)
            % 将当前坐标轴更改为 handles.effimg 所代表的坐标轴，清空坐标轴并显示处理后的图像
            % 更新 handles 结构
            guidata(hObject, handles);
            % 显示等待条
            h = waitbar(0, '更新图像中 等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            % 更新函数
            update(app,handles);
        end
        % f1_3 : 均值滤波 circular
        function f1_3_Callback(app, event)
            % 创建 GUIDE 风格的回调函数参数（由迁移工具添加）
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>
            % 显示等待条
            h = waitbar(0, '等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            mean_size = [app.global_MeanFilter, app.global_MeanFilter];
            % 使用 convertToGUIDECallbackArguments 函数将 app 和 event 转换为 GUIDE 回调函数所需的参数，并将其存储在 hObject、eventdata 和 handles 变量中
            h = fspecial('average', mean_size);
            % 创建一个平均滤波器模板
            handles.img=imfilter(handles.img,h,'circular');
            % 使用平均滤波器对图像进行滤波处理
            axes(handles.effimg);
            cla;
            imshow(handles.img)
            % 将当前坐标轴更改为 handles.effimg 所代表的坐标轴，清空坐标轴并显示处理后的图像
            % 更新 handles 结构
            guidata(hObject, handles);
            % 显示等待条
            h = waitbar(0, '更新图像中 等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            % 更新函数
            update(app,handles);
        end
        % Value changed function: EditField_MeanFilterChanged
        function EditField_MeanFilterChanged(app, event)
            app.global_MeanFilter = app.EditField_MeanFilter.Value;
        end
        
        % f3 中值滤波
        function f3_Callback(app, event)
            % 定义一个名为 f3_Callback 的函数，其输入参数为 app 和 event
            % 创建 GUIDE 风格的回调函数参数（由迁移工具添加）
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>
            % 显示等待条
            h = waitbar(0, '等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            median_size = [app.global_MedianFilter, app.global_MedianFilter];
            % 使用 convertToGUIDECallbackArguments 函数将 app 和 event 转换为 GUIDE 回调函数所需的参数，并将其存储在 hObject、eventdata 和 handles 变量中
            mysize=size(handles.img);             % 获取处理后的图像大小
            if numel(mysize)>2
                r=medfilt2(handles.img(:,:,1), median_size);
                % 对图像的红色通道进行中值滤波处理
                g=medfilt2(handles.img(:,:,2), median_size);
                % 对图像的绿色通道进行中值滤波处理
                b=medfilt2(handles.img(:,:,3), median_size);
                % 对图像的蓝色通道进行中值滤波处理
                handles.img=cat(3,r,g,b);
            else
                gray=medfilt2(handles.img(:,:,1), median_size);
                handles.img=cat(1, gray);
            end
            
            % 将三个滤波处理后的通道重新组合成图像
            axes(handles.effimg);
            % 将当前坐标轴更改为 handles.effimg 所代表的坐标轴
            cla; imshow(handles.img);
            % 清空坐标轴并显示处理后的图像
            guidata(hObject,handles);
            % 更新应用程序数据，将 handles 变量保存到 app 中
            % 更新 handles 结构
            guidata(hObject, handles);
            % 显示等待条
            h = waitbar(0, '更新图像中 等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            % 更新函数
            update(app,handles);
        end
        % Value changed function: EditField_MedianFilterChanged
        function EditField_MedianFilterChanged(app, event)
            app.global_MedianFilter = app.EditField_MedianFilter.Value;
        end
        
        %非局部均值去噪
        function  f4_Callback(app, event)
            % 从输入参数中获取对象句柄和其他信息
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>
            
            % 显示等待条
            h_waitbar = waitbar(0, '等待... 正在计算,非局部均值去噪计算时间较长，请耐心等待。');
            steps = 200;
            for step = 0:100
                waitbar(step / steps)
            end
            % 获取图像及相关参数
            img = handles.img; % 获取图像
            %I:含噪声图像
            %ds:邻域窗口半径
            %Ds:搜索窗口半径
            %h:高斯函数平滑参数
            %DenoisedImg：去噪图像
            h = app.global_nlmeansH; % 获取非局部均值去噪参数h
            ds = app.global_patchSize; % 获取像素块大小参数
            Ds = app.global_searchWindowSize; % 获取搜索窗口大小参数
            
            I=double(img);
            [m,n]=size(I);
            DenoisedImg=zeros(m,n);
            PaddedImg = padarray(I,[ds,ds],'symmetric','both');
            kernel=ones(2*ds+1,2*ds+1);
            kernel=kernel./((2*ds+1)*(2*ds+1));
            h2=h*h;
            for i=1:m
                for j=1:n
                    i1=i+ds;
                    j1=j+ds;
                    W1=PaddedImg(i1-ds:i1+ds,j1-ds:j1+ds);%邻域窗口1
                    wmax=0;
                    average=0;
                    sweight=0;
                    %%搜索窗口
                    rmin = max(i1-Ds,ds+1);
                    rmax = min(i1+Ds,m+ds);
                    smin = max(j1-Ds,ds+1);
                    smax = min(j1+Ds,n+ds);
                    for r=rmin:rmax
                        for s=smin:smax
                            if(r==i1&&s==j1)
                                continue;
                            end
                            W2=PaddedImg(r-ds:r+ds,s-ds:s+ds);%邻域窗口2
                            Dist2=sum(sum(kernel.*(W1-W2).*(W1-W2)));%邻域间距离
                            w=exp(-Dist2/h2);
                            if(w>wmax)
                                wmax=w;
                            end
                            sweight=sweight+w;
                            average=average+w*PaddedImg(r,s);
                        end
                    end
                    average=average+wmax*PaddedImg(i1,j1);%自身取最大权值
                    sweight=sweight+wmax;
                    DenoisedImg(i,j)=average/sweight;
                end
            end
            
            for step = 100:200
                waitbar(step / steps)
            end
            close(h_waitbar)
            % 更新图像句柄中的图像数据
            handles.img = DenoisedImg;
            assignin('base', 'myVariable2', handles.img);
            % 在图像窗口中显示处理后的图像
            axes(handles.effimg);
            cla;
            imshow(handles.img,[]);
            
            % 更新图像句柄
            guidata(hObject, handles);
            
            % 更新应用程序
            update(app,handles);
        end
        
        % Value changed function: EditField_lambdaChanged
        function EditField_nlmeansHChanged(app, event)
            app.global_nlmeansH = app.EditField_nlmeansH.Value;
        end
        % Value changed function: EditField_numIterationsChanged
        function EditField_patchSizeChanged(app, event)
            app.global_patchSize = app.EditField_patchSize.Value;
        end
        % Value changed function: EditField_deltaTChanged
        function EditField_searchWindowSizeChanged(app, event)
            app.global_searchWindowSize = app.EditField_searchWindowSize.Value;
        end
        
        %总变差去噪
        function f5_Callback(app, event)
            % 创建 GUIDE 风格的回调函数参数
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>
            
            % 显示等待条
            h = waitbar(0, '等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            
            % 获取处理后的图像大小
            mysize = size(handles.img);
            if numel(mysize) > 2
                % 对图像的三个通道分别进行总变差去噪处理
                r = total_variation_denoising(app, handles.img(:,:,1)); % 对红色通道进行总变差去噪处理
                g = total_variation_denoising(app, handles.img(:,:,2)); % 对绿色通道进行总变差去噪处理
                b = total_variation_denoising(app, handles.img(:,:,3)); % 对蓝色通道进行总变差去噪处理
                handles.img = cat(3, r, g, b);
            else
                % 对灰度图像进行总变差去噪处理
                gray = total_variation_denoising(app, handles.img(:,:,1));
                handles.img = cat(1, gray);
            end
            
            % 将处理后的图像显示在坐标轴上
            axes(handles.effimg);
            cla; imshow(handles.img,[]);
            
            % 更新应用程序数据
            guidata(hObject, handles);
            
            % 显示等待条
            h = waitbar(0, '更新图像中 等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            
            % 更新函数
            update(app, handles);
        end
        function denoised_img = total_variation_denoising(app, img)
            % 设置总变差去噪参数
            lambda = 0.1; % 正则化参数，控制去噪程度
            num_iterations = 30; % 迭代次数
            delta_t = 0.1; % 时间步长
            
            % 将输入图像转换为双精度数组
            img = double(img);
            
            % 初始化
            denoised_img = img;
            
            % 迭代优化
            for i = 1:num_iterations
                % 计算梯度
                [gx, gy] = gradient(denoised_img);
                
                % 计算梯度的模长
                grad_norm = sqrt(gx.^2 + gy.^2);
                
                % 计算梯度的方向
                grad_dir = atan2(gy, gx);
                
                % 更新像素值
                for x = 2:size(denoised_img, 1)-1
                    for y = 2:size(denoised_img, 2)-1
                        % 计算局部梯度的散度
                        div = (grad_norm(x, y) - grad_norm(x-1, y) + grad_norm(x, y) - grad_norm(x, y-1));
                        % 更新像素值
                        denoised_img(x, y) = denoised_img(x, y) + delta_t * lambda * div;
                    end
                end
            end
        end
        function div = divergence(app, grad_dir, grad_norm)
            % 计算梯度的散度
            [gx, gy] = pol2cart(grad_dir, grad_norm);
            div = divergence(gx, gy);
        end
        % Value changed function: EditField_lambdaChanged
        function EditField_lambdaChanged(app, event)
            app.global_lambda = app.EditField_lambda.Value;
        end
        % Value changed function: EditField_numIterationsChanged
        function EditField_numIterationsChanged(app, event)
            app.global_numIterations = app.EditField_numIterations.Value;
        end
        % Value changed function: EditField_deltaTChanged
        function EditField_deltaTChanged(app, event)
            app.global_deltaT = app.EditField_deltaT.Value;
        end
        
        % Button pushed function: n1 高斯噪声
        function n1_Callback(app, event)
            % Create GUIDE-style callback args - Added by Migration Tool
            [hObject, ~, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>
            
            % 显示等待条
            h = waitbar(0, '等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            
            % 指定均值和方差
            mean_value = app.globalGau_MEAN;  % 均值
            variance_value = app.globalGau_VAR;  % 方差
            
            % 向图像添加高斯噪声
            handles.img = imnoise(handles.img, 'gaussian', mean_value, variance_value);
            
            % 在orgimg中显示带噪声的图像
            axes(handles.effimg);
            cla;
            imshow(handles.img);
            
            % 更新 handles 结构
            guidata(hObject, handles);
            
            % 显示等待条
            h = waitbar(0, '更新图像中 等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            
            % 更新函数
            update(app,handles);
        end
        
        % Button pushed function: n2 泊松噪声
        function n2_Callback(app, event)
            % Create GUIDE-style callback args - Added by Migration Tool
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>
            % 显示等待条
            h = waitbar(0, '等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            % 向图像添加泊松噪声
            handles.img = imnoise(handles.img, 'poisson');
            % 在orgimg中显示带噪声的图像
            axes(handles.effimg);
            cla;
            imshow(handles.img);
            % 更新 handles 结构
            guidata(hObject, handles);
            % 显示等待条
            h = waitbar(0, '更新图像中 等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            % 更新函数
            update(app,handles);
        end
        
        % 按钮触发函数：n3_Callback 添加椒盐噪声
        function n3_Callback(app, event)
            % 显示等待条
            h = waitbar(0, '等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h);
            % 创建 GUIDE 风格的回调函数参数（由迁移工具添加）
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>
            %读取噪声密度
            density = app.globalDENSITY;
            % 使用 convertToGUIDECallbackArguments 函数将 app 和 event 转换为 GUIDE 回调函数所需的参数，并将其存储在 hObject、eventdata 和 handles 变量中
            handles.img = imnoise(handles.img,'salt & pepper',density);
            % 对图像添加椒盐噪声，噪声密度为density
            axes(handles.effimg);
            % 将当前坐标轴更改为 handles.orgimg 所代表的坐标轴
            cla; imshow(handles.img);
            % 清空坐标轴并显示处理后的图像
            % 更新 handles 结构
            guidata(hObject, handles);
            % 显示等待条
            h = waitbar(0, '更新图像中 等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h);
            % 更新函数
            update(app,handles);
        end
        
        % Button pushed function: n4 斑点噪声
        function n4_Callback(app, event)
            % 创建 GUIDE 风格的回调参数 - 迁移工具添加
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>
            % 显示等待条
            h = waitbar(0, '等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            % 指定斑点噪声的方差
            speckle_variance = app.global_SPEVAR;
            % 对图像进行斑点噪声处理
            handles.img = imnoise(handles.img, 'speckle', speckle_variance);
            % 在 "effimg" axes 中显示处理后的图像
            axes(handles.effimg);
            cla;
            imshow(handles.img);
            % 更新 handles 结构
            guidata(hObject, handles);
            % 显示等待条
            h = waitbar(0, '更新图像中 等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            % 更新函数
            update(app,handles);
        end
        % Value changed function: EditField_idealp_FreChanged
        function EditField_SPEVARChanged(app, event)
            app.global_SPEVAR = app.EditField_SPEVAR.Value;
        end
        
        % Button pushed function: n5 运动噪声
        function n5_Callback(app, event)
            % 创建 GUIDE 风格的回调参数 - 迁移工具添加
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>
            % 显示等待条
            h = waitbar(0, '等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            set(handles.p1, 'Enable', 'on');
            
            I = handles.img; % 原始图像
            I = im2double(I); % 转换为双精度类型
            LEN = app.globalVar_LEN; % 参数设置 线性运动的长度
            THETA = app.globalVar_THETA; % 参数设置 线性运动的方向角度
            PSF = fspecial('motion', LEN, THETA); % 产生点扩散函数PSF
            J = imfilter(I, PSF, 'conv', 'circular'); % 进行运动模糊处理
            noise = 0.03 * randn(size(I)); % 生成噪声
            K = imadd(J, noise); % 将噪声添加到模糊图像中
            K = im2uint8(K); % 转换为 uint8 类型
            axes(handles.effimg);
            cla;
            imshow(K);
            handles.img = K;
            % 更新 handles 结构
            guidata(hObject, handles);
            
            NP = abs(fft2(noise)).^2; % 计算噪声功率谱密度
            NPower = sum(NP(:)) / prod(size(noise));
            NCORR = fftshift(real(ifft2(NP))); % 计算噪声功率谱密度的自相关
            IP = abs(fft2(I).^2); % 计算图像功率谱密度
            IPower = sum(IP(:)) / prod(size(I));
            ICORR = fftshift(real(ifft2(IP))); % 计算图像功率谱密度的自相关
            app.globalVar_PSF = PSF;
            app.globalVar_NCORR = NCORR;
            app.globalVar_ICORR = ICORR;
            app.globalVar_K = K;
            
            % 显示等待条
            h = waitbar(0, '更新图像中 等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            % 更新函数
            update(app,handles);
            close(h)
        end
        % Value changed function: EditField_LEN
        function EditField_LENChanged(app, event)
            app.globalVar_LEN = app.EditField_LEN.Value;
        end
        % Value changed function: EditField_THETA
        function EditField_THETAChanged(app, event)
            app.globalVar_THETA = app.EditField_THETA.Value;
        end
        
        % Button pushed function: p1 维纳滤波
        function p1_Callback(app, event)
            % Create GUIDE-style callback args - Added by Migration Tool
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>
            % 显示等待条
            h = waitbar(0, '等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            
            PSF = app.globalVar_PSF;
            NCORR = app.globalVar_NCORR;
            ICORR = app.globalVar_ICORR;
            K = app.globalVar_K;
            L = deconvwnr(K, PSF, NCORR, ICORR); % 使用维纳滤波进行复原处理
            
            handles.img = L;
            % 更新 handles 结构
            guidata(hObject, handles);
            axes(handles.effimg);
            cla;
            imshow(L);
            % 显示等待条
            h = waitbar(0, '更新图像中 等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            % 更新函数
            update(app,handles);
            close(h)
        end
        
        %理想低通滤波器
        function p21_Callback(app, event)
            % Create GUIDE-style callback args - Added by Migration Tool
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>
            % 显示等待条
            h = waitbar(0, '等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            % 理想低通滤波器
            mysize = size(handles.img); % 获取图像大小
            if numel(mysize) > 2 % 如果图像为彩色图像
                handles.img = rgb2gray(handles.img); % 转换为灰度图像
            end
            
            I = handles.img; % 原始图像
            I = im2double(I); % 转换为双精度类型
            M = 2 * size(I, 1); % 滤波器行数
            N = 2 * size(I, 2); % 滤波器列数
            u = -M/2:(M/2-1);
            v = -N/2:(N/2-1);
            [U, V] = meshgrid(u, v);
            D = sqrt(U.^2 + V.^2);
            D0 = app.global_idealp_Fre; %低通截止频率
            H = double(D <= D0); % 计算理想低通滤波器的传递函数
            J = fftshift(fft2(I, size(H, 1), size(H, 2))); % 对输入图像进行傅里叶变换，并将零频率移动到频谱中心
            K = J .* H; % 进行频域滤波
            L = ifft2(ifftshift(K)); % 对滤波后的频谱进行傅里叶反变换
            L = L(1:size(I, 1), 1:size(I, 2)); % 提取出与原始图像大小相同的部分作为输出结果
            handles.img = L;
            
            axes(handles.effimg);
            cla;
            imshow(handles.img);
            % 更新 handles 结构
            guidata(hObject, handles);
            % 显示等待条
            h = waitbar(0, '更新图像中 等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            % 更新函数
            update(app,handles);
        end
        % Value changed function: EditField_idealp_FreChanged
        function EditField_idealp_FreChanged(app, event)
            app.global_idealp_Fre = app.EditField_idealp_Fre.Value;
        end
        %理想高通滤波器
        function p22_Callback(app, event)
            % Create GUIDE-style callback args - Added by Migration Tool
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>
            % 显示等待条
            h = waitbar(0, '等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            % 理想高通滤波器
            mysize = size(handles.img); % 获取图像大小
            if numel(mysize) > 2 % 如果图像为彩色图像
                handles.img = rgb2gray(handles.img); % 转换为灰度图像
            end
            
            I = handles.img; % 原始图像
            I = im2double(I); % 转换为双精度类型
            M = 2 * size(I, 1); % 滤波器行数
            N = 2 * size(I, 2); % 滤波器列数
            u = -M/2:(M/2-1);
            v = -N/2:(N/2-1);
            [U, V] = meshgrid(u, v);
            D = sqrt(U.^2 + V.^2);
            D0 = app.global_ideahp_Fre;
            H_lowpass = double(D <= D0); % 计算理想低通滤波器的传递函数
            H_highpass = 1 - H_lowpass; % 计算理想高通滤波器的传递函数
            J = fftshift(fft2(I, size(H_highpass, 1), size(H_highpass, 2))); % 对输入图像进行傅里叶变换，并将零频率移动到频谱中心
            K = J .* H_highpass; % 进行频域滤波
            L = ifft2(ifftshift(K)); % 对滤波后的频谱进行傅里叶反变换
            L = L(1:size(I, 1), 1:size(I, 2)); % 提取出与原始图像大小相同的部分作为输出结果
            handles.img = L;
            
            axes(handles.effimg);
            cla;
            imshow(handles.img);
            % 更新 handles 结构
            guidata(hObject, handles);
            % 显示等待条
            h = waitbar(0, '更新图像中 等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            % 更新函数
            update(app,handles);
        end
        % Value changed function: EditField_ideahp_FreChanged
        function EditField_ideahp_FreChanged(app, event)
            app.global_ideahp_Fre = app.EditField_ideahp_Fre.Value;
        end
        %理想带通滤波器
        function p23_Callback(app, event)
            % Create GUIDE-style callback args - Added by Migration Tool
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>
            % 显示等待条
            h = waitbar(0, '等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            % 理想带通滤波器
            mysize = size(handles.img); % 获取图像大小
            if numel(mysize) > 2 % 如果图像为彩色图像
                handles.img = rgb2gray(handles.img); % 转换为灰度图像
            end
            
            I = handles.img; % 原始图像
            I = im2double(I); % 转换为双精度类型
            M = 2 * size(I, 1); % 滤波器行数
            N = 2 * size(I, 2); % 滤波器列数
            u = -M/2:(M/2-1);
            v = -N/2:(N/2-1);
            [U, V] = meshgrid(u, v);
            D = sqrt(U.^2 + V.^2);
            D0_low = 20;
            D0_high = 80;
            % 计算理想带通滤波器的传递函数
            H = double((D >= D0_low) & (D <= D0_high));
            J = fftshift(fft2(I, size(H, 1), size(H, 2))); % 对输入图像进行傅里叶变换，并将零频率移动到频谱中心
            K = J .* H; % 进行频域滤波
            L = ifft2(ifftshift(K)); % 对滤波后的频谱进行傅里叶反变换
            L = L(1:size(I, 1), 1:size(I, 2)); % 提取出与原始图像大小相同的部分作为输出结果
            handles.img = L;
            
            axes(handles.effimg);
            cla;
            imshow(handles.img);
            % 更新 handles 结构
            guidata(hObject, handles);
            % 显示等待条
            h = waitbar(0, '更新图像中 等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            % 更新函数
            update(app,handles);
        end
        % Value changed function: EditField_ideabp_lFreChanged
        function EditField_ideabp_lFreChanged(app, event)
            app.global_ideabp_lFre = app.EditField_ideabp_lFre.Value;
        end
        % Value changed function: EditField_ideabp_hFreChanged
        function EditField_ideabp_hFreChanged(app, event)
            app.global_ideabp_hFre = app.EditField_ideabp_hFre.Value;
        end
        %理想带阻滤波器
        function p24_Callback(app, event)
            % Create GUIDE-style callback args - Added by Migration Tool
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>
            % 显示等待条
            h = waitbar(0, '等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            % 带阻滤波器
            mysize = size(handles.img); % 获取图像大小
            if numel(mysize) > 2 % 如果图像为彩色图像
                handles.img = rgb2gray(handles.img); % 转换为灰度图像
            end
            
            I = handles.img; % 原始图像
            I = im2double(I); % 转换为双精度类型
            M = 2 * size(I, 1); % 滤波器行数
            N = 2 * size(I, 2); % 滤波器列数
            u = -M/2:(M/2-1);
            v = -N/2:(N/2-1);
            [U, V] = meshgrid(u, v);
            D = sqrt(U.^2 + V.^2);
            D0 = app.global_ideabs_Fre; % 滤波器D0
            W = app.global_ideabs_w; % 滤波器带宽
            H = double(or(D < (D0 - W / 2), D > D0 + W / 2)); % 计算带阻滤波器的传递函数
            J = fftshift(fft2(I, size(H, 1), size(H, 2))); % 对输入图像进行傅里叶变换，并将零频率移动到频谱中心
            K = J .* H; % 进行频域滤波
            L = ifft2(ifftshift(K)); % 对滤波后的频谱进行傅里叶反变换
            L = L(1:size(I, 1), 1:size(I, 2)); % 提取出与原始图像大小相同的部分作为输出结果
            handles.img = L;
            
            axes(handles.effimg);
            cla;
            imshow(handles.img);
            % 更新 handles 结构
            guidata(hObject, handles);
            % 显示等待条
            h = waitbar(0, '更新图像中 等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            % 更新函数
            update(app,handles);
        end
        % Value changed function: EditField_ideabs_FreChanged
        function EditField_ideabs_FreChanged(app, event)
            app.global_ideabs_Fre = app.EditField_ideabs_Fre.Value;
        end
        % Value changed function: EditField_ideabs_wChanged
        function EditField_ideabs_wChanged(app, event)
            app.global_ideabs_w = app.EditField_ideabs_w.Value;
        end
        
        % Value changed function: EditField_butterOrderChanged
        function EditField_butterOrderChanged(app, event)
            app.global_butterOrder = app.EditField_butterOrder.Value;
        end
        %巴特沃斯低通滤波器
        function p31_Callback(app, event)
            % Create GUIDE-style callback args - Added by Migration Tool
            [hObject, ~, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>
            % 显示等待条
            h = waitbar(0, '等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            % 巴特沃斯低通滤波器
            mysize = size(handles.img); % 获取图像大小
            if numel(mysize) > 2 % 如果图像为彩色图像
                handles.img = rgb2gray(handles.img); % 转换为灰度图像
            end
            
            I = handles.img; % 原始图像
            I = im2double(I); % 转换为双精度类型
            M = 2 * size(I, 1); % 滤波器行数
            N = 2 * size(I, 2); % 滤波器列数
            u = -M/2:(M/2-1);
            v = -N/2:(N/2-1);
            [U, V] = meshgrid(u, v);
            D = sqrt(U.^2 + V.^2);
            D0 = 60; % 截止频率
            n = app.global_butterOrder; % 巴特沃斯滤波器阶数
            H = 1 ./ (1 + (D ./ D0).^(2*n)); % 计算巴特沃斯低通滤波器的传递函数
            J = fftshift(fft2(I, size(H, 1), size(H, 2))); % 对输入图像进行傅里叶变换，并将零频率移动到频谱中心
            K = J .* H; % 进行频域滤波
            L = ifft2(ifftshift(K)); % 对滤波后的频谱进行傅里叶反变换
            L = L(1:size(I, 1), 1:size(I, 2)); % 提取出与原始图像大小相同的部分作为输出结果
            handles.img = L;
            
            axes(handles.effimg);
            cla;
            imshow(handles.img,[]);
            % 更新 handles 结构
            guidata(hObject, handles);
            % 显示等待条
            h = waitbar(0, '更新图像中 等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            % 更新函数
            update(app,handles);
        end
        % Value changed function: EditField_idealp_FreChanged
        function EditField_butterlp_FreChanged(app, event)
            app.global_butterlp_Fre = app.EditField_butterlp_Fre.Value;
        end
        %巴特沃斯高通滤波器
        function p32_Callback(app, event)
            % Create GUIDE-style callback args - Added by Migration Tool
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>
            % 显示等待条
            h = waitbar(0, '等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            % 巴特沃斯高通滤波器
            mysize = size(handles.img); % 获取图像大小
            if numel(mysize) > 2 % 如果图像为彩色图像
                handles.img = rgb2gray(handles.img); % 转换为灰度图像
            end
            
            I = handles.img; % 原始图像
            I = im2double(I); % 转换为双精度类型
            M = 2 * size(I, 1); % 滤波器行数
            N = 2 * size(I, 2); % 滤波器列数
            u = -M/2:(M/2-1);
            v = -N/2:(N/2-1);
            [U, V] = meshgrid(u, v);
            D = sqrt(U.^2 + V.^2);
            D0 = app.global_butterhp_Fre; % 截止频率
            n = app.global_butterOrder; % 巴特沃斯滤波器阶数
            H = 1 ./ (1 + (D0 ./ D).^(2*n)); % 计算巴特沃斯高通滤波器的传递函数
            J = fftshift(fft2(I, size(H, 1), size(H, 2))); % 对输入图像进行傅里叶变换，并将零频率移动到频谱中心
            K = J .* H; % 进行频域滤波
            L = ifft2(ifftshift(K)); % 对滤波后的频谱进行傅里叶反变换
            L = L(1:size(I, 1), 1:size(I, 2)); % 提取出与原始图像大小相同的部分作为输出结果
            handles.img = L;
            
            axes(handles.effimg);
            cla;
            imshow(handles.img,[]);
            % 更新 handles 结构
            guidata(hObject, handles);
            % 显示等待条
            h = waitbar(0, '更新图像中 等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            % 更新函数
            update(app,handles);
        end
        % Value changed function: EditField_idealp_FreChanged
        function EditField_butterhp_FreChanged(app, event)
            app.global_butterhp_Fre = app.EditField_butterhp_Fre.Value;
        end
        %巴特沃斯带通滤波器
        function p33_Callback(app, event)
            % Create GUIDE-style callback args - Added by Migration Tool
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>
            
            % 显示等待条
            h = waitbar(0, '等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            
            % 巴特沃斯带通滤波器
            mysize = size(handles.img); % 获取图像大小
            if numel(mysize) > 2 % 如果图像为彩色图像
                handles.img = rgb2gray(handles.img); % 转换为灰度图像
            end
            
            I = mat2gray(handles.img,[0 255]);; % 原始图像
            M = 2 * size(I, 1); % 滤波器行数
            N = 2 * size(I, 2); % 滤波器列数
            u = -M/2:(M/2-1);
            v = -N/2:(N/2-1);
            [U, V] = meshgrid(u, v);
            D = sqrt(U.^2 + V.^2);
            D0_low = app.global_butterbp_lFre; % 低频截止频率
            D0_high = app.global_butterbp_hFre; % 高频截止频率
            n = app.global_butterOrder; % 巴特沃斯滤波器阶数
            H_low = 1 ./ (1 + (D ./ D0_low).^(2*n)); % 低通滤波器传递函数
            H_high = 1 ./ (1 + (D0_high ./ D).^(2*n)); % 高通滤波器传递函数
            H_bandpass = H_high .* H_low; % 带通滤波器传递函数
            J = fftshift(fft2(I, size(H_bandpass, 1), size(H_bandpass, 2))); % 对输入图像进行傅里叶变换，并将零频率移动到频谱中心
            K = J .* H_bandpass; % 进行频域滤波
            L = ifft2(ifftshift(K)); % 对滤波后的频谱进行傅里叶反变换
            L = L(1:size(I, 1), 1:size(I, 2)); % 提取出与原始图像大小相同的部分作为输出结果
            handles.img = L;
            
            axes(handles.effimg);
            cla;
            imshow(handles.img,[]);
            
            % 更新 handles 结构
            guidata(hObject, handles);
            
            % 显示等待条
            h = waitbar(0, '更新图像中 等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            
            % 更新函数
            update(app,handles);
        end
        % Value changed function: EditField_idealp_FreChanged
        function EditField_butterbp_lFreChanged(app, event)
            app.global_butterbp_lFre = app.EditField_butterbp_lFre.Value;
        end
        % Value changed function: EditField_idealp_FreChanged
        function EditField_butterbp_hFreChanged(app, event)
            app.global_butterbp_hFre = app.EditField_butterbp_hFre.Value;
        end
        % 巴特沃斯带阻滤波器
        function p34_Callback(app, event)
            % Create GUIDE-style callback args - Added by Migration Tool
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>
            
            % 显示等待条
            h = waitbar(0, '等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            
            % 巴特沃斯带阻滤波器
            mysize = size(handles.img); % 获取图像大小
            if numel(mysize) > 2 % 如果图像为彩色图像
                handles.img = rgb2gray(handles.img); % 转换为灰度图像
            end
            
            I = mat2gray(handles.img,[0 255]); % 原始图像
            M = 2 * size(I, 1); % 滤波器行数
            N = 2 * size(I, 2); % 滤波器列数
            u = -M/2:(M/2-1);
            v = -N/2:(N/2-1);
            [U, V] = meshgrid(u, v);
            D = sqrt(U.^2 + V.^2);
            D_width = app.global_butterbs_w;
            D_fre = app.global_butterbs_Fre;
            D0_low = D_fre - D_width/2; % 低频截止频率
            D0_high = D_fre + D_width/2; % 高频截止频率
            n = app.global_butterOrder; % 巴特沃斯滤波器阶数
            H_low = 1 ./ (1 + (D0_low ./ D).^(2*n)); % 低通滤波器传递函数
            H_high = 1 ./ (1 + (D ./ D0_high).^(2*n)); % 高通滤波器传递函数
            H_bandstop = H_high .* H_low; % 带阻滤波器传递函数
            J = fftshift(fft2(I, size(H_bandstop, 1), size(H_bandstop, 2))); % 对输入图像进行傅里叶变换，并将零频率移动到频谱中心
            K = J .* H_bandstop; % 进行频域滤波
            L = ifft2(ifftshift(K)); % 对滤波后的频谱进行傅里叶反变换
            L = L(1:size(I, 1), 1:size(I, 2)); % 提取出与原始图像大小相同的部分作为输出结果
            handles.img = L;
            
            axes(handles.effimg);
            cla;
            imshow(handles.img,[]);
            
            % 更新 handles 结构
            guidata(hObject, handles);
            
            % 显示等待条
            h = waitbar(0, '更新图像中 等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            
            % 更新函数
            update(app,handles);
        end
        % Value changed function: EditField_idealp_FreChanged
        function EditField_butterbs_FreChanged(app, event)
            app.global_butterbs_Fre = app.EditField_butterbs_Fre.Value;
        end
        % Value changed function: EditField_idealp_FreChanged
        function EditField_butterbs_wChanged(app, event)
            app.global_butterbs_w = app.EditField_butterbs_w.Value;
        end
        
        % f21 高斯低通滤波
        function f21_Callback(app, event)
            % 创建 GUIDE 风格的回调函数参数（由迁移工具添加）
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event);
            % 显示等待条
            h = waitbar(0, '等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            % 使用 convertToGUIDECallbackArguments 函数将 app 和 event 转换为 GUIDE 回调函数所需的参数，并将其存储在 hObject、eventdata 和 handles 变量中
            D0 = app.global_gausslp_Fre; % 截止频率设置在半径值为50处
            original_img = mat2gray(handles.img,[0 255]);
            filtered_img = GaussianLowpass(app, original_img, D0);
            handles.img = filtered_img;
            % 显示图像
            axes(handles.effimg); cla; imshow(handles.img,[]);
            % 将当前坐标轴更改为 handles.effimg 所代表的坐标轴，清空坐标轴并显示处理后的图像
            % 更新 handles 结构
            guidata(hObject, handles);
            % 显示等待条
            h = waitbar(0, '更新图像中 等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            % 更新函数
            update(app, handles);
        end
        function img_filtered = GaussianLowpass(app, img, D0)
            [M, N] = size(img);
            P = 2 * M;
            Q = 2 * N;
            fc = zeros(M, N);
            
            for x = 1:M
                for y = 1:N
                    fc(x, y) = img(x, y) * (-1)^(x+y);
                end
            end
            
            F = fft2(fc, P, Q);
            H = zeros(P, Q);
            
            for x = (-P/2):P/2-1
                for y = (-Q/2):Q/2-1
                    D = sqrt(x^2 + y^2);
                    H(x+(P/2)+1, y+(Q/2)+1) = exp(-(D^2)/(2*D0^2));
                end
            end
            
            G = H .* F;
            
            g = real(ifft2(G));
            g = g(1:1:M, 1:1:N);
            
            img_filtered = zeros(M, N);
            
            for x = 1:1:M
                for y = 1:1:N
                    img_filtered(x, y) = g(x, y) * (-1)^(x+y);
                end
            end
            
        end
        % Value changed function: EditField_idealp_FreChanged
        function EditField_gausslp_FreChanged(app, event)
            app.global_gausslp_Fre = app.EditField_gausslp_Fre.Value;
        end
        % f22 高斯高通滤波
        function f22_Callback(app, event)
            % 创建 GUIDE 风格的回调函数参数（由迁移工具添加）
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event);
            % 显示等待条
            h = waitbar(0, '等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            % 使用 convertToGUIDECallbackArguments 函数将 app 和 event 转换为 GUIDE 回调函数所需的参数，并将其存储在 hObject、eventdata 和 handles 变量中
            D0 = app.global_gausshp_Fre; % 截止频率设置在半径值为50处
            original_img = mat2gray(handles.img,[0 255]);
            filtered_img = GaussianHighpass(app, original_img, D0);
            handles.img = filtered_img;
            % 显示图像
            axes(handles.effimg); cla; imshow(handles.img,[]);
            % 将当前坐标轴更改为 handles.effimg 所代表的坐标轴，清空坐标轴并显示处理后的图像
            % 更新 handles 结构
            guidata(hObject, handles);
            % 显示等待条
            h = waitbar(0, '更新图像中 等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            % 更新函数
            update(app, handles);
        end
        function img_filtered = GaussianHighpass(app, img, D0)
            [M, N] = size(img);
            P = 2 * M;
            Q = 2 * N;
            fc = zeros(M, N);
            
            for x = 1:M
                for y = 1:N
                    fc(x, y) = img(x, y) * (-1)^(x+y);
                end
            end
            
            F = fft2(fc, P, Q);
            H = zeros(P, Q);
            
            for x = (-P/2):P/2-1
                for y = (-Q/2):Q/2-1
                    D = sqrt(x^2 + y^2);
                    H(x+(P/2)+1, y+(Q/2)+1) = 1 - exp(-(D^2)/(2*D0^2)); % 使用 1 减去低通滤波器的传递函数来获取高通滤波器
                end
            end
            
            G = H .* F;
            
            g = real(ifft2(G));
            g = g(1:1:M, 1:1:N);
            
            img_filtered = zeros(M, N);
            
            for x = 1:1:M
                for y = 1:1:N
                    img_filtered(x, y) = g(x, y) * (-1)^(x+y);
                end
            end
            
        end
        % Value changed function: EditField_idealp_FreChanged
        function EditField_gausshp_FreChanged(app, event)
            app.global_gausshp_Fre = app.EditField_gausshp_Fre.Value;
        end
        % f23 高斯带通滤波
        function f23_Callback(app, event)
            % 创建 GUIDE 风格的回调函数参数（由迁移工具添加）
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event);
            % 显示等待条
            h = waitbar(0, '等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            % 使用 convertToGUIDECallbackArguments 函数将 app 和 event 转换为 GUIDE 回调函数所需的参数，并将其存储在 hObject、eventdata 和 handles 变量中
            D0_low = app.global_gaussbp_lFre; % 低频截止频率
            D0_high = app.global_gaussbp_hFre; % 高频截止频率
            original_img = mat2gray(handles.img,[0 255]);
            filtered_img = GaussianBandpass(app, original_img, D0_low, D0_high);
            handles.img = filtered_img;
            % 显示图像
            axes(handles.effimg); cla; imshow(handles.img,[]);
            % 将当前坐标轴更改为 handles.effimg 所代表的坐标轴，清空坐标轴并显示处理后的图像
            % 更新 handles 结构
            guidata(hObject, handles);
            % 显示等待条
            h = waitbar(0, '更新图像中 等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            % 更新函数
            update(app, handles);
        end
        function img_filtered = GaussianBandpass(app, img, D0_low, D0_high)
            [M, N] = size(img);
            P = 2 * M;
            Q = 2 * N;
            fc = zeros(M, N);
            
            for x = 1:M
                for y = 1:N
                    fc(x, y) = img(x, y) * (-1)^(x+y);
                end
            end
            
            F = fft2(fc, P, Q);
            H = zeros(P, Q);
            
            for x = (-P/2):P/2-1
                for y = (-Q/2):Q/2-1
                    D = sqrt(x^2 + y^2);
                    H(x+(P/2)+1, y+(Q/2)+1) = (1 - exp(-(D^2)/(2*D0_low^2))) .* (exp(-(D^2)/(2*D0_high^2))); % 使用高斯函数得到带通滤波器的传递函数
                end
            end
            
            G = H .* F;
            
            g = real(ifft2(G));
            g = g(1:1:M, 1:1:N);
            
            img_filtered = zeros(M, N);
            
            for x = 1:1:M
                for y = 1:1:N
                    img_filtered(x, y) = g(x, y) * (-1)^(x+y);
                end
            end
            
        end
        % Value changed function: EditField_idealp_FreChanged
        function EditField_gaussbp_lFreChanged(app, event)
            app.global_gaussbp_lFre = app.EditField_gaussbp_lFre.Value;
        end
        % Value changed function: EditField_idealp_FreChanged
        function EditField_gaussbp_hFreChanged(app, event)
            app.global_gaussbp_hFre = app.EditField_gaussbp_hFre.Value;
        end
        % f23 高斯带阻滤波
        function f24_Callback(app, event)
            % 创建 GUIDE 风格的回调函数参数（由迁移工具添加）
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event);
            % 显示等待条
            h = waitbar(0, '等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            % 使用 convertToGUIDECallbackArguments 函数将 app 和 event 转换为 GUIDE 回调函数所需的参数，并将其存储在 hObject、eventdata 和 handles 变量中
            D_width = app.global_gaussbs_w;
            D_fre = app.global_gaussbs_Fre;
            D0_low = D_fre - D_width/2; % 低频截止频率
            D0_high = D_fre + D_width/2; % 高频截止频率
            original_img = mat2gray(handles.img,[0 255]);
            filtered_img = GaussianBandstop(app, original_img, D0_low, D0_high);
            handles.img = filtered_img;
            % 显示图像
            axes(handles.effimg); cla; imshow(handles.img,[]);
            % 将当前坐标轴更改为 handles.effimg 所代表的坐标轴，清空坐标轴并显示处理后的图像
            % 更新 handles 结构
            guidata(hObject, handles);
            % 显示等待条
            h = waitbar(0, '更新图像中 等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            % 更新函数
            update(app, handles);
        end
        function img_filtered = GaussianBandstop(app, img, D0_low, D0_high)
            [M, N] = size(img);
            P = 2 * M;
            Q = 2 * N;
            fc = zeros(M, N);
            
            for x = 1:M
                for y = 1:N
                    fc(x, y) = img(x, y) * (-1)^(x+y);
                end
            end
            
            F = fft2(fc, P, Q);
            H = zeros(P, Q);
            
            for x = (-P/2):P/2-1
                for y = (-Q/2):Q/2-1
                    D = sqrt(x^2 + y^2);
                    H(x+(P/2)+1, y+(Q/2)+1) = 1 - exp(-(D^2)/(2*D0_high^2)) .* (1 - exp(-(D^2)/(2*D0_low^2))); % 使用高斯函数得到带阻滤波器的传递函数
                end
            end
            
            G = H .* F;
            
            g = real(ifft2(G));
            g = g(1:1:M, 1:1:N);
            
            img_filtered = zeros(M, N);
            
            for x = 1:1:M
                for y = 1:1:N
                    img_filtered(x, y) = g(x, y) * (-1)^(x+y);
                end
            end
            
        end
        % Value changed function: EditField_idealp_FreChanged
        function EditField_gaussbs_FreChanged(app, event)
            app.global_gaussbs_Fre = app.EditField_gaussbs_Fre.Value;
        end
        % Value changed function: EditField_idealp_FreChanged
        function EditField_gaussbs_wChanged(app, event)
            app.global_gaussbs_w = app.EditField_gaussbs_w.Value;
        end
        
        %小波去噪
        function p4_Callback(app, event)
            % 创建 GUIDE 风格的回调参数 - 迁移工具添加
            [hObject, ~, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>
            
            % 获取处理后的图像
            if numel(size(handles.img)) > 2
                grayimg = rgb2gray(handles.img);
            else
                grayimg = handles.img;
            end
            
            % 小波变换
            [LLY, ~, ~, ~] = dwt2(grayimg, 'haar');
            
            % 小波去噪处理
            % 这里可以根据具体情况选择不同的去噪方法，比如基于阈值的去噪方法
            threshold = app.global_wtThreshold; % 设定阈值
            
            % 改进的小波去噪方法：基于软阈值的小波去噪
            % 调整阈值值
            threshold = threshold * std(LLY(:)); % 根据小波系数的标准差动态调整阈值
            
            LLY_denoised = wthresh(LLY, 's', threshold);
            
            % 逆小波变换恢复图像
            denoised_img = idwt2(LLY_denoised, [], [], [], 'haar');
            
            handles.img = denoised_img;
            % 显示去噪后的图像
            axes(handles.effimg);
            cla;
            imshow(handles.img, []);
            
            % 显示等待条
            h = waitbar(0, '更新图像中 等待...');
            steps = 100;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            % 更新函数
            update(app, handles);
        end
        % Value changed function: EditField_THETA
        function EditField_wtThresholdChanged(app, event)
            app.global_wtThreshold = app.EditField_wtThreshold.Value;
        end
        
        % Button pushed function: p5 同态滤波器
        function p5_Callback(app, event)
            % Create GUIDE-style callback args - Added by Migration Tool
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>
            % 显示等待条
            h = waitbar(0, '等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            % 同态滤波器
            % 检查图像通道数，如果是彩色图像则转换为灰度图像
            mysize = size(handles.img); % 获取图像的大小
            if numel(mysize) > 2 % 如果图像为彩色图像
                handles.img = rgb2gray(handles.img); % 转换为灰度图像
            end
            
            I = handles.img; % 原始图像
            J = log(im2double(I) + 1); % 对原始图像进行对数变换
            K = fft2(J); % 进行二维傅里叶变换
            
            % 参数设置
            n = app.global_homeOrder; % 阶数
            D0 = app.global_Fre * pi; % 截止频率
            rh = app.global_hGain; % 高通增益
            rl = app.global_lGain; % 低通增益
            
            % 创建同态滤波器
            [row, column] = size(J); % 获取图像的行列数
            [X, Y] = meshgrid(1:column, 1:row); % 创建X, Y网格
            D1 = sqrt((X - round(row/2)).^2 + (Y - round(column/2)).^2); % 计算频率距离
            H = rl + (rh ./ (1 + (D0./D1).^(2*n))); % 计算同态滤波器H
            
            L = K .* H; % 滤波器与频域图像相乘
            M = ifft2(L); % 进行反傅里叶变换
            N = exp(M) - 1; % 恢复图像
            N = real(N); % 取实部
            handles.img = double(N);
            % 更新图像
            axes(handles.effimg);
            cla;
            imshow(handles.img);
            % 更新 handles 结构
            guidata(hObject, handles);
            % 显示等待条
            h = waitbar(0, '更新图像中 等待...');
            steps = 200;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
            % 更新函数
            update(app,handles);
        end
        % Value changed function: EditField_homeOrder
        function EditField_homeOrderChanged(app, event)
            app.global_homeOrder = app.EditField_homeOrder.Value;
        end
        % Value changed function: EditField_Fre
        function EditField_FreChanged(app, event)
            app.global_Fre = app.EditField_Fre.Value;
        end
        % Value changed function: EditField_hGain
        function EditField_hGainChanged(app, event)
            app.global_hGain = app.EditField_hGain.Value;
        end
        % Value changed function: EditField_lGain
        function EditField_lGainChanged(app, event)
            app.global_lGain = app.EditField_lGain.Value;
        end
        
        % Button pushed function: Button1 小波变换haar
        function Button1Pushed(app, event)
            % 计算小波变换
            % 创建 GUIDE 风格的回调参数 - 迁移工具添加
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>
            mysize=size(handles.img);             % 获取处理后的图像大小
            if numel(mysize)>2
                grayimg = rgb2gray(handles.img);
            else
                grayimg = handles.img;
            end
            [grap_LLY, HL, LH, HH] = dwt2(grayimg, 'haar');
            [LLY, HL, LH, HH] = dwt2(handles.img, 'haar');
            %显示小波变换
            set(handles.g4, 'Visible', 'on');
            axes(handles.g4);
            cla;
            imshow(grap_LLY, []);
            hold off;
            title('小波变换(1.灰度低频近似值)');
            set(handles.g5, 'Visible', 'on');
            axes(handles.g5);
            cla;
            imshow(HL, []);
            title('2.水平方向细节');
            hold off;
            set(handles.g6, 'Visible', 'on');
            axes(handles.g6);
            cla;
            imshow(LH, []);
            title('3.垂直方向细节');
            hold off;
            set(handles.g7, 'Visible', 'on');
            axes(handles.g7);
            cla;
            imshow(HH, []);
            title('4.对角线方向细节');
            hold off;
            % 显示等待条
            h = waitbar(0, '更新图像中 等待...');
            steps = 100;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
        end
        
        % Button pushed function: Button2 小波变换db2
        function Button2Pushed(app, event)
            % 创建 GUIDE 风格的回调参数 - 迁移工具添加
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>
            % 计算小波变换
            mysize=size(handles.img);             % 获取处理后的图像大小
            if numel(mysize)>2
                grayimg = rgb2gray(handles.img);
            else
                grayimg = handles.img;
            end
            [grap_LLY, HL, LH, HH] = dwt2(grayimg, 'db2');
            [LLY, HL, LH, HH] = dwt2(handles.img, 'db2');
            %显示小波变换
            set(handles.g4, 'Visible', 'on');
            axes(handles.g4);
            cla;
            imshow(grap_LLY, []);
            hold off;
            title('小波变换(1.灰度低频近似值)');
            set(handles.g5, 'Visible', 'on');
            axes(handles.g5);
            cla;
            imshow(HL, []);
            title('2.水平方向细节');
            hold off;
            set(handles.g6, 'Visible', 'on');
            axes(handles.g6);
            cla;
            imshow(LH, []);
            title('3.垂直方向细节');
            hold off;
            set(handles.g7, 'Visible', 'on');
            axes(handles.g7);
            cla;
            imshow(HH, []);
            title('4.对角线方向细节');
            hold off;
            % 显示等待条
            h = waitbar(0, '更新图像中 等待...');
            steps = 100;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
        end
        
        % Button pushed function: Button3 小波变换bior1.1
        function Button3Pushed(app, event)
            % 创建 GUIDE 风格的回调参数 - 迁移工具添加
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>
            % 计算小波变换
            mysize=size(handles.img);             % 获取处理后的图像大小
            if numel(mysize)>2
                grayimg = rgb2gray(handles.img);
            else
                grayimg = handles.img;
            end
            [grap_LLY, HL, LH, HH] = dwt2(grayimg, 'bior1.1');
            [LLY, HL, LH, HH] = dwt2(handles.img, 'bior1.1');
            %显示小波变换
            set(handles.g4, 'Visible', 'on');
            axes(handles.g4);
            cla;
            imshow(grap_LLY, []);
            hold off;
            title('小波变换(1.灰度低频近似值)');
            set(handles.g5, 'Visible', 'on');
            axes(handles.g5);
            cla;
            imshow(HL, []);
            title('2.水平方向细节');
            hold off;
            set(handles.g6, 'Visible', 'on');
            axes(handles.g6);
            cla;
            imshow(LH, []);
            title('3.垂直方向细节');
            hold off;
            set(handles.g7, 'Visible', 'on');
            axes(handles.g7);
            cla;
            imshow(HH, []);
            title('4.对角线方向细节');
            hold off;
            % 显示等待条
            h = waitbar(0, '更新图像中 等待...');
            steps = 100;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
        end
        
        % Button pushed function: Button4 小波变换coif1
        function Button4Pushed(app, event)
            % 创建 GUIDE 风格的回调参数 - 迁移工具添加
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>
            % 计算小波变换
            mysize=size(handles.img);             % 获取处理后的图像大小
            if numel(mysize)>2
                grayimg = rgb2gray(handles.img);
            else
                grayimg = handles.img;
            end
            [grap_LLY, HL, LH, HH] = dwt2(grayimg, 'coif1');
            [LLY, HL, LH, HH] = dwt2(handles.img, 'coif1');
            %显示小波变换
            set(handles.g4, 'Visible', 'on');
            axes(handles.g4);
            cla;
            imshow(grap_LLY, []);
            hold off;
            title('小波变换(1.灰度低频近似值)');
            set(handles.g5, 'Visible', 'on');
            axes(handles.g5);
            cla;
            imshow(HL, []);
            title('2.水平方向细节');
            hold off;
            set(handles.g6, 'Visible', 'on');
            axes(handles.g6);
            cla;
            imshow(LH, []);
            title('3.垂直方向细节');
            hold off;
            set(handles.g7, 'Visible', 'on');
            axes(handles.g7);
            cla;
            imshow(HH, []);
            title('4.对角线方向细节');
            hold off;
            % 显示等待条
            h = waitbar(0, '更新图像中 等待...');
            steps = 100;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
        end
        
        % Button pushed function: Button5 小波变换sym2
        function Button5Pushed(app, event)
            % 创建 GUIDE 风格的回调参数 - 迁移工具添加
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>
            % 计算小波变换
            mysize=size(handles.img);             % 获取处理后的图像大小
            if numel(mysize)>2
                grayimg = rgb2gray(handles.img);
            else
                grayimg = handles.img;
            end
            [grap_LLY, HL, LH, HH] = dwt2(grayimg, 'sym2');
            [LLY, HL, LH, HH] = dwt2(handles.img, 'sym2');
            %显示小波变换
            set(handles.g4, 'Visible', 'on');
            axes(handles.g4);
            cla;
            imshow(grap_LLY, []);
            hold off;
            title('小波变换(1.灰度低频近似值)');
            set(handles.g5, 'Visible', 'on');
            axes(handles.g5);
            cla;
            imshow(HL, []);
            title('2.水平方向细节');
            hold off;
            set(handles.g6, 'Visible', 'on');
            axes(handles.g6);
            cla;
            imshow(LH, []);
            title('3.垂直方向细节');
            hold off;
            set(handles.g7, 'Visible', 'on');
            axes(handles.g7);
            cla;
            imshow(HH, []);
            title('4.对角线方向细节');
            hold off;
            % 显示等待条
            h = waitbar(0, '更新图像中 等待...');
            steps = 100;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
        end
        
        % Button pushed function: Button6 小波变换fk4
        function Button6Pushed(app, event)
            % 创建 GUIDE 风格的回调参数 - 迁移工具添加
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>
            % 计算小波变换
            mysize=size(handles.img);             % 获取处理后的图像大小
            if numel(mysize)>2
                grayimg = rgb2gray(handles.img);
            else
                grayimg = handles.img;
            end
            [grap_LLY, HL, LH, HH] = dwt2(grayimg, 'fk4');
            [LLY, HL, LH, HH] = dwt2(handles.img, 'fk4');
            %显示小波变换
            set(handles.g4, 'Visible', 'on');
            axes(handles.g4);
            cla;
            imshow(grap_LLY, []);
            hold off;
            title('小波变换(1.灰度低频近似值)');
            set(handles.g5, 'Visible', 'on');
            axes(handles.g5);
            cla;
            imshow(HL, []);
            title('2.水平方向细节');
            hold off;
            set(handles.g6, 'Visible', 'on');
            axes(handles.g6);
            cla;
            imshow(LH, []);
            title('3.垂直方向细节');
            hold off;
            set(handles.g7, 'Visible', 'on');
            axes(handles.g7);
            cla;
            imshow(HH, []);
            title('4.对角线方向细节');
            hold off;
            % 显示等待条
            h = waitbar(0, '更新图像中 等待...');
            steps = 100;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
        end
        
        % Button pushed function: Button7 小波变换dmey
        function Button7Pushed(app, event)
            % 创建 GUIDE 风格的回调参数 - 迁移工具添加
            [hObject, eventdata, handles] = convertToGUIDECallbackArguments(app, event); %#ok<ASGLU>
            % 计算小波变换
            mysize=size(handles.img);             % 获取处理后的图像大小
            if numel(mysize)>2
                grayimg = rgb2gray(handles.img);
            else
                grayimg = handles.img;
            end
            [grap_LLY, HL, LH, HH] = dwt2(grayimg, 'dmey');
            [LLY, HL, LH, HH] = dwt2(handles.img, 'dmey');
            %显示小波变换
            set(handles.g4, 'Visible', 'on');
            axes(handles.g4);
            cla;
            imshow(grap_LLY, []);
            hold off;
            title('小波变换(1.灰度低频近似值)');
            set(handles.g5, 'Visible', 'on');
            axes(handles.g5);
            cla;
            imshow(HL, []);
            title('2.水平方向细节');
            hold off;
            set(handles.g6, 'Visible', 'on');
            axes(handles.g6);
            cla;
            imshow(LH, []);
            title('3.垂直方向细节');
            hold off;
            set(handles.g7, 'Visible', 'on');
            axes(handles.g7);
            cla;
            imshow(HH, []);
            title('4.对角线方向细节');
            hold off;
            % 显示等待条
            h = waitbar(0, '更新图像中 等待...');
            steps = 100;
            for step = 1:steps
                waitbar(step / steps)
            end
            close(h)
        end
        
        
        
        % Value changed function: EditField_MEAN
        function EditField_MEANChanged(app, event)
            app.globalGau_MEAN = app.EditField_MEAN.Value;
        end
        
        % Value changed function: EditField_VARChanged
        function EditField_VARChanged(app, event)
            app.globalGau_VAR = app.EditField_VAR.Value;
        end
        
        % Value changed function: EditField_DENSITYChanged
        function EditField_DENSITYChanged(app, event)
            app.globalDENSITY = app.EditField_DENSITY.Value;
        end
        
    end
    
    % UI界面布局初始化
    methods (Access = private)
        
        % 创建 UIFigure 和组件
        function createComponents(app)
            
            %创建组件
            try
                % 创建图形并隐藏，直到创建所有组件
                app.figure = uifigure('Visible', 'off');
                app.figure.Colormap = [0.2431 0.149 0.6588;0.2431 0.1529 0.6745;0.2471 0.1569 0.6863;0.2471 0.1608 0.698;0.251 0.1647 0.7059;0.251 0.1686 0.7176;0.2549 0.1725 0.7294;0.2549 0.1765 0.7412;0.2588 0.1804 0.749;0.2588 0.1843 0.7608;0.2627 0.1922 0.7843;0.2627 0.1961 0.7922;0.2667 0.2 0.8039;0.2667 0.2039 0.8157;0.2706 0.2078 0.8235;0.2706 0.2157 0.8353;0.2706 0.2196 0.8431;0.2745 0.2235 0.851;0.2745 0.2275 0.8627;0.2745 0.2314 0.8706;0.2745 0.2392 0.8784;0.2784 0.2431 0.8824;0.2784 0.2471 0.8902;0.2784 0.2549 0.898;0.2784 0.2588 0.902;0.2784 0.2667 0.9098;0.2784 0.2706 0.9137;0.2784 0.2745 0.9216;0.2824 0.2824 0.9255;0.2824 0.2863 0.9294;0.2824 0.2941 0.9333;0.2824 0.298 0.9412;0.2824 0.3059 0.9451;0.2824 0.3098 0.949;0.2824 0.3137 0.9529;0.2824 0.3216 0.9569;0.2824 0.3255 0.9608;0.2824 0.3294 0.9647;0.2784 0.3373 0.9686;0.2784 0.3412 0.9686;0.2784 0.349 0.9725;0.2784 0.3529 0.9765;0.2784 0.3569 0.9804;0.2784 0.3647 0.9804;0.2745 0.3686 0.9843;0.2745 0.3765 0.9843;0.2745 0.3804 0.9882;0.2706 0.3843 0.9882;0.2706 0.3922 0.9922;0.2667 0.3961 0.9922;0.2627 0.4039 0.9922;0.2627 0.4078 0.9961;0.2588 0.4157 0.9961;0.2549 0.4196 0.9961;0.251 0.4275 0.9961;0.2471 0.4314 1;0.2431 0.4392 1;0.2353 0.4431 1;0.2314 0.451 1;0.2235 0.4549 1;0.2196 0.4627 0.9961;0.2118 0.4667 0.9961;0.2078 0.4745 0.9922;0.2 0.4784 0.9922;0.1961 0.4863 0.9882;0.1922 0.4902 0.9882;0.1882 0.498 0.9843;0.1843 0.502 0.9804;0.1843 0.5098 0.9804;0.1804 0.5137 0.9765;0.1804 0.5176 0.9725;0.1804 0.5255 0.9725;0.1804 0.5294 0.9686;0.1765 0.5333 0.9647;0.1765 0.5412 0.9608;0.1765 0.5451 0.9569;0.1765 0.549 0.9529;0.1765 0.5569 0.949;0.1725 0.5608 0.9451;0.1725 0.5647 0.9412;0.1686 0.5686 0.9373;0.1647 0.5765 0.9333;0.1608 0.5804 0.9294;0.1569 0.5843 0.9255;0.1529 0.5922 0.9216;0.1529 0.5961 0.9176;0.149 0.6 0.9137;0.149 0.6039 0.9098;0.1451 0.6078 0.9098;0.1451 0.6118 0.9059;0.1412 0.6196 0.902;0.1412 0.6235 0.898;0.1373 0.6275 0.898;0.1373 0.6314 0.8941;0.1333 0.6353 0.8941;0.1294 0.6392 0.8902;0.1255 0.6471 0.8902;0.1216 0.651 0.8863;0.1176 0.6549 0.8824;0.1137 0.6588 0.8824;0.1137 0.6627 0.8784;0.1098 0.6667 0.8745;0.1059 0.6706 0.8706;0.102 0.6745 0.8667;0.098 0.6784 0.8627;0.0902 0.6824 0.8549;0.0863 0.6863 0.851;0.0784 0.6902 0.8471;0.0706 0.6941 0.8392;0.0627 0.698 0.8353;0.0549 0.702 0.8314;0.0431 0.702 0.8235;0.0314 0.7059 0.8196;0.0235 0.7098 0.8118;0.0157 0.7137 0.8078;0.0078 0.7176 0.8;0.0039 0.7176 0.7922;0 0.7216 0.7882;0 0.7255 0.7804;0 0.7294 0.7765;0.0039 0.7294 0.7686;0.0078 0.7333 0.7608;0.0157 0.7333 0.7569;0.0235 0.7373 0.749;0.0353 0.7412 0.7412;0.051 0.7412 0.7373;0.0627 0.7451 0.7294;0.0784 0.7451 0.7216;0.0902 0.749 0.7137;0.102 0.7529 0.7098;0.1137 0.7529 0.702;0.1255 0.7569 0.6941;0.1373 0.7569 0.6863;0.1451 0.7608 0.6824;0.1529 0.7608 0.6745;0.1608 0.7647 0.6667;0.1686 0.7647 0.6588;0.1725 0.7686 0.651;0.1804 0.7686 0.6471;0.1843 0.7725 0.6392;0.1922 0.7725 0.6314;0.1961 0.7765 0.6235;0.2 0.7804 0.6157;0.2078 0.7804 0.6078;0.2118 0.7843 0.6;0.2196 0.7843 0.5882;0.2235 0.7882 0.5804;0.2314 0.7882 0.5725;0.2392 0.7922 0.5647;0.251 0.7922 0.5529;0.2588 0.7922 0.5451;0.2706 0.7961 0.5373;0.2824 0.7961 0.5255;0.2941 0.7961 0.5176;0.3059 0.8 0.5059;0.3176 0.8 0.498;0.3294 0.8 0.4863;0.3412 0.8 0.4784;0.3529 0.8 0.4667;0.3686 0.8039 0.4549;0.3804 0.8039 0.4471;0.3922 0.8039 0.4353;0.4039 0.8039 0.4235;0.4196 0.8039 0.4118;0.4314 0.8039 0.4;0.4471 0.8039 0.3922;0.4627 0.8 0.3804;0.4745 0.8 0.3686;0.4902 0.8 0.3569;0.5059 0.8 0.349;0.5176 0.8 0.3373;0.5333 0.7961 0.3255;0.5451 0.7961 0.3176;0.5608 0.7961 0.3059;0.5765 0.7922 0.2941;0.5882 0.7922 0.2824;0.6039 0.7882 0.2745;0.6157 0.7882 0.2627;0.6314 0.7843 0.251;0.6431 0.7843 0.2431;0.6549 0.7804 0.2314;0.6706 0.7804 0.2235;0.6824 0.7765 0.2157;0.698 0.7765 0.2078;0.7098 0.7725 0.2;0.7216 0.7686 0.1922;0.7333 0.7686 0.1843;0.7451 0.7647 0.1765;0.7608 0.7647 0.1725;0.7725 0.7608 0.1647;0.7843 0.7569 0.1608;0.7961 0.7569 0.1569;0.8078 0.7529 0.1529;0.8157 0.749 0.1529;0.8275 0.749 0.1529;0.8392 0.7451 0.1529;0.851 0.7451 0.1569;0.8588 0.7412 0.1569;0.8706 0.7373 0.1608;0.8824 0.7373 0.1647;0.8902 0.7373 0.1686;0.902 0.7333 0.1765;0.9098 0.7333 0.1804;0.9176 0.7294 0.1882;0.9255 0.7294 0.1961;0.9373 0.7294 0.2078;0.9451 0.7294 0.2157;0.9529 0.7294 0.2235;0.9608 0.7294 0.2314;0.9686 0.7294 0.2392;0.9765 0.7294 0.2431;0.9843 0.7333 0.2431;0.9882 0.7373 0.2431;0.9961 0.7412 0.2392;0.9961 0.7451 0.2353;0.9961 0.7529 0.2314;0.9961 0.7569 0.2275;0.9961 0.7608 0.2235;0.9961 0.7686 0.2196;0.9961 0.7725 0.2157;0.9961 0.7804 0.2078;0.9961 0.7843 0.2039;0.9961 0.7922 0.2;0.9922 0.7961 0.1961;0.9922 0.8039 0.1922;0.9922 0.8078 0.1922;0.9882 0.8157 0.1882;0.9843 0.8235 0.1843;0.9843 0.8275 0.1804;0.9804 0.8353 0.1804;0.9765 0.8392 0.1765;0.9765 0.8471 0.1725;0.9725 0.851 0.1686;0.9686 0.8588 0.1647;0.9686 0.8667 0.1647;0.9647 0.8706 0.1608;0.9647 0.8784 0.1569;0.9608 0.8824 0.1569;0.9608 0.8902 0.1529;0.9608 0.898 0.149;0.9608 0.902 0.149;0.9608 0.9098 0.1451;0.9608 0.9137 0.1412;0.9608 0.9216 0.1373;0.9608 0.9255 0.1333;0.9608 0.9333 0.1294;0.9647 0.9373 0.1255;0.9647 0.9451 0.1216;0.9647 0.949 0.1176;0.9686 0.9569 0.1098;0.9686 0.9608 0.1059;0.9725 0.9686 0.102;0.9725 0.9725 0.0941;0.9765 0.9765 0.0863;0.9765 0.9843 0.0824];
                app.figure.Position = [50 60 1620 850];
                app.figure.Name = 'Image_processing_GUI';
                app.figure.HandleVisibility = 'callback';
                app.figure.Tag = 'figure';
            catch
                errordlg('发生了错误，请检查输入或操作','错误','modal');
            end
            
            %文件操作控件
            try
                % Create uibuttongroup1
                app.uibuttongroup1 = uibuttongroup(app.figure);
                app.uibuttongroup1.Title = '文件操作';
                app.uibuttongroup1.Tag = 'uibuttongroup1';
                app.uibuttongroup1.FontName = 'Microsoft YaHei UI';
                app.uibuttongroup1.FontSize = 16;
                app.uibuttongroup1.Position = [29 492 102 300];
                
                % Create Label_Name
                app.Label_Name = uilabel(app.figure);
                app.Label_Name.FontWeight = 'bold';
                app.Label_Name.FontColor = [0.118 0.565 1]; % 深海蓝色
                app.Label_Name.Position = [18 400 400 100];
                app.Label_Name.FontName = '宋体';
                app.Label_Name.FontSize = 18;
                app.Label_Name.Text = '  姓名';
                
                % Create Label_Number
                app.Label_Number = uilabel(app.figure);
                app.Label_Number.FontWeight = 'bold';
                app.Label_Number.FontColor = [0.118 0.565 1]; % 深海蓝色
                app.Label_Number.Position = [18 360 400 100];
                app.Label_Number.FontName = '宋体';
                app.Label_Number.FontSize = 18;
                app.Label_Number.Text = '  学号';
            catch
                errordlg('发生了错误，请检查输入或操作','错误','modal');
            end
            
            % Create load
            try
                app.load = uibutton(app.uibuttongroup1, 'push');
                app.load.ButtonPushedFcn = createCallbackFcn(app, @load_Callback, true);
                app.load.Tag = 'load';
                app.load.FontName = 'Microsoft YaHei UI';
                app.load.FontSize = 20;
                app.load.Position = [18 211 64 42];
                app.load.Text = '打开';
            catch
                errordlg('发生了错误，请检查输入或操作','错误','modal');
            end
            
            % Create exit
            try
                app.exit = uibutton(app.uibuttongroup1, 'push');
                app.exit.ButtonPushedFcn = createCallbackFcn(app, @exit_Callback, true);
                app.exit.Tag = 'exit';
                app.exit.FontName = 'Microsoft YaHei UI';
                app.exit.FontSize = 20;
                app.exit.Position = [18 154 64 42];
                app.exit.Text = '退出';
            catch
                errordlg('发生了错误，请检查输入或操作','错误','modal');
            end
            
            % Create save
            try
                app.save = uibutton(app.uibuttongroup1, 'push');
                app.save.ButtonPushedFcn = createCallbackFcn(app, @save_Callback, true);
                app.save.Tag = 'save';
                app.save.FontName = 'Microsoft YaHei UI';
                app.save.FontSize = 20;
                app.save.Position = [18 91 64 42];
                app.save.Text = '保存';
            catch
                errordlg('发生了错误，请检查输入或操作','错误','modal');
            end
            
            % Create reset
            try
                app.reset = uibutton(app.uibuttongroup1, 'push');
                app.reset.ButtonPushedFcn = createCallbackFcn(app, @reset_Callback, true);
                app.reset.Tag = 'reset';
                app.reset.FontName = 'Microsoft YaHei UI';
                app.reset.FontSize = 20;
                app.reset.Position = [18 30 64 42];
                app.reset.Text = '清除';
            catch
                errordlg('发生了错误，请检查输入或操作','错误','modal');
            end
            % Create uibuttongroup2 原始图像
            try
                app.uibuttongroup2 = uibuttongroup(app.figure);
                app.uibuttongroup2.Title = '原始图像';
                app.uibuttongroup2.Tag = 'uibuttongroup2';
                app.uibuttongroup2.FontName = 'Microsoft YaHei UI';
                app.uibuttongroup2.FontSize = 24;
                app.uibuttongroup2.Position = [150 400 390 435];
                % Create orgimg
                app.orgimg = uiaxes(app.uibuttongroup2);
                app.orgimg.FontName = 'Microsoft YaHei UI';
                app.orgimg.Colormap = [0.2431 0.149 0.6588;0.2431 0.1529 0.6745;0.2471 0.1569 0.6863;0.2471 0.1608 0.698;0.251 0.1647 0.7059;0.251 0.1686 0.7176;0.2549 0.1725 0.7294;0.2549 0.1765 0.7412;0.2588 0.1804 0.749;0.2588 0.1843 0.7608;0.2627 0.1922 0.7843;0.2627 0.1961 0.7922;0.2667 0.2 0.8039;0.2667 0.2039 0.8157;0.2706 0.2078 0.8235;0.2706 0.2157 0.8353;0.2706 0.2196 0.8431;0.2745 0.2235 0.851;0.2745 0.2275 0.8627;0.2745 0.2314 0.8706;0.2745 0.2392 0.8784;0.2784 0.2431 0.8824;0.2784 0.2471 0.8902;0.2784 0.2549 0.898;0.2784 0.2588 0.902;0.2784 0.2667 0.9098;0.2784 0.2706 0.9137;0.2784 0.2745 0.9216;0.2824 0.2824 0.9255;0.2824 0.2863 0.9294;0.2824 0.2941 0.9333;0.2824 0.298 0.9412;0.2824 0.3059 0.9451;0.2824 0.3098 0.949;0.2824 0.3137 0.9529;0.2824 0.3216 0.9569;0.2824 0.3255 0.9608;0.2824 0.3294 0.9647;0.2784 0.3373 0.9686;0.2784 0.3412 0.9686;0.2784 0.349 0.9725;0.2784 0.3529 0.9765;0.2784 0.3569 0.9804;0.2784 0.3647 0.9804;0.2745 0.3686 0.9843;0.2745 0.3765 0.9843;0.2745 0.3804 0.9882;0.2706 0.3843 0.9882;0.2706 0.3922 0.9922;0.2667 0.3961 0.9922;0.2627 0.4039 0.9922;0.2627 0.4078 0.9961;0.2588 0.4157 0.9961;0.2549 0.4196 0.9961;0.251 0.4275 0.9961;0.2471 0.4314 1;0.2431 0.4392 1;0.2353 0.4431 1;0.2314 0.451 1;0.2235 0.4549 1;0.2196 0.4627 0.9961;0.2118 0.4667 0.9961;0.2078 0.4745 0.9922;0.2 0.4784 0.9922;0.1961 0.4863 0.9882;0.1922 0.4902 0.9882;0.1882 0.498 0.9843;0.1843 0.502 0.9804;0.1843 0.5098 0.9804;0.1804 0.5137 0.9765;0.1804 0.5176 0.9725;0.1804 0.5255 0.9725;0.1804 0.5294 0.9686;0.1765 0.5333 0.9647;0.1765 0.5412 0.9608;0.1765 0.5451 0.9569;0.1765 0.549 0.9529;0.1765 0.5569 0.949;0.1725 0.5608 0.9451;0.1725 0.5647 0.9412;0.1686 0.5686 0.9373;0.1647 0.5765 0.9333;0.1608 0.5804 0.9294;0.1569 0.5843 0.9255;0.1529 0.5922 0.9216;0.1529 0.5961 0.9176;0.149 0.6 0.9137;0.149 0.6039 0.9098;0.1451 0.6078 0.9098;0.1451 0.6118 0.9059;0.1412 0.6196 0.902;0.1412 0.6235 0.898;0.1373 0.6275 0.898;0.1373 0.6314 0.8941;0.1333 0.6353 0.8941;0.1294 0.6392 0.8902;0.1255 0.6471 0.8902;0.1216 0.651 0.8863;0.1176 0.6549 0.8824;0.1137 0.6588 0.8824;0.1137 0.6627 0.8784;0.1098 0.6667 0.8745;0.1059 0.6706 0.8706;0.102 0.6745 0.8667;0.098 0.6784 0.8627;0.0902 0.6824 0.8549;0.0863 0.6863 0.851;0.0784 0.6902 0.8471;0.0706 0.6941 0.8392;0.0627 0.698 0.8353;0.0549 0.702 0.8314;0.0431 0.702 0.8235;0.0314 0.7059 0.8196;0.0235 0.7098 0.8118;0.0157 0.7137 0.8078;0.0078 0.7176 0.8;0.0039 0.7176 0.7922;0 0.7216 0.7882;0 0.7255 0.7804;0 0.7294 0.7765;0.0039 0.7294 0.7686;0.0078 0.7333 0.7608;0.0157 0.7333 0.7569;0.0235 0.7373 0.749;0.0353 0.7412 0.7412;0.051 0.7412 0.7373;0.0627 0.7451 0.7294;0.0784 0.7451 0.7216;0.0902 0.749 0.7137;0.102 0.7529 0.7098;0.1137 0.7529 0.702;0.1255 0.7569 0.6941;0.1373 0.7569 0.6863;0.1451 0.7608 0.6824;0.1529 0.7608 0.6745;0.1608 0.7647 0.6667;0.1686 0.7647 0.6588;0.1725 0.7686 0.651;0.1804 0.7686 0.6471;0.1843 0.7725 0.6392;0.1922 0.7725 0.6314;0.1961 0.7765 0.6235;0.2 0.7804 0.6157;0.2078 0.7804 0.6078;0.2118 0.7843 0.6;0.2196 0.7843 0.5882;0.2235 0.7882 0.5804;0.2314 0.7882 0.5725;0.2392 0.7922 0.5647;0.251 0.7922 0.5529;0.2588 0.7922 0.5451;0.2706 0.7961 0.5373;0.2824 0.7961 0.5255;0.2941 0.7961 0.5176;0.3059 0.8 0.5059;0.3176 0.8 0.498;0.3294 0.8 0.4863;0.3412 0.8 0.4784;0.3529 0.8 0.4667;0.3686 0.8039 0.4549;0.3804 0.8039 0.4471;0.3922 0.8039 0.4353;0.4039 0.8039 0.4235;0.4196 0.8039 0.4118;0.4314 0.8039 0.4;0.4471 0.8039 0.3922;0.4627 0.8 0.3804;0.4745 0.8 0.3686;0.4902 0.8 0.3569;0.5059 0.8 0.349;0.5176 0.8 0.3373;0.5333 0.7961 0.3255;0.5451 0.7961 0.3176;0.5608 0.7961 0.3059;0.5765 0.7922 0.2941;0.5882 0.7922 0.2824;0.6039 0.7882 0.2745;0.6157 0.7882 0.2627;0.6314 0.7843 0.251;0.6431 0.7843 0.2431;0.6549 0.7804 0.2314;0.6706 0.7804 0.2235;0.6824 0.7765 0.2157;0.698 0.7765 0.2078;0.7098 0.7725 0.2;0.7216 0.7686 0.1922;0.7333 0.7686 0.1843;0.7451 0.7647 0.1765;0.7608 0.7647 0.1725;0.7725 0.7608 0.1647;0.7843 0.7569 0.1608;0.7961 0.7569 0.1569;0.8078 0.7529 0.1529;0.8157 0.749 0.1529;0.8275 0.749 0.1529;0.8392 0.7451 0.1529;0.851 0.7451 0.1569;0.8588 0.7412 0.1569;0.8706 0.7373 0.1608;0.8824 0.7373 0.1647;0.8902 0.7373 0.1686;0.902 0.7333 0.1765;0.9098 0.7333 0.1804;0.9176 0.7294 0.1882;0.9255 0.7294 0.1961;0.9373 0.7294 0.2078;0.9451 0.7294 0.2157;0.9529 0.7294 0.2235;0.9608 0.7294 0.2314;0.9686 0.7294 0.2392;0.9765 0.7294 0.2431;0.9843 0.7333 0.2431;0.9882 0.7373 0.2431;0.9961 0.7412 0.2392;0.9961 0.7451 0.2353;0.9961 0.7529 0.2314;0.9961 0.7569 0.2275;0.9961 0.7608 0.2235;0.9961 0.7686 0.2196;0.9961 0.7725 0.2157;0.9961 0.7804 0.2078;0.9961 0.7843 0.2039;0.9961 0.7922 0.2;0.9922 0.7961 0.1961;0.9922 0.8039 0.1922;0.9922 0.8078 0.1922;0.9882 0.8157 0.1882;0.9843 0.8235 0.1843;0.9843 0.8275 0.1804;0.9804 0.8353 0.1804;0.9765 0.8392 0.1765;0.9765 0.8471 0.1725;0.9725 0.851 0.1686;0.9686 0.8588 0.1647;0.9686 0.8667 0.1647;0.9647 0.8706 0.1608;0.9647 0.8784 0.1569;0.9608 0.8824 0.1569;0.9608 0.8902 0.1529;0.9608 0.898 0.149;0.9608 0.902 0.149;0.9608 0.9098 0.1451;0.9608 0.9137 0.1412;0.9608 0.9216 0.1373;0.9608 0.9255 0.1333;0.9608 0.9333 0.1294;0.9647 0.9373 0.1255;0.9647 0.9451 0.1216;0.9647 0.949 0.1176;0.9686 0.9569 0.1098;0.9686 0.9608 0.1059;0.9725 0.9686 0.102;0.9725 0.9725 0.0941;0.9765 0.9765 0.0863;0.9765 0.9843 0.0824];
                app.orgimg.FontSize = 13;
                app.orgimg.NextPlot = 'replace';
                app.orgimg.Tag = 'orgimg';
                app.orgimg.Position = [0 30 320 320];
            catch
                errordlg('发生了错误，请检查输入或操作','错误','modal');
            end
            % Create uibuttongroup3 修改后图像
            try
                app.uibuttongroup3 = uibuttongroup(app.figure);
                app.uibuttongroup3.Title = '效果预览图像';
                app.uibuttongroup3.Tag = 'uibuttongroup3';
                app.uibuttongroup3.FontName = 'Microsoft YaHei UI';
                app.uibuttongroup3.FontSize = 24;
                app.uibuttongroup3.Position = [550 400 390 435];
                % Create effimg
                app.effimg = uiaxes(app.uibuttongroup3);
                app.effimg.FontName = 'Microsoft YaHei UI';
                app.effimg.Colormap = [0.2431 0.149 0.6588;0.2431 0.1529 0.6745;0.2471 0.1569 0.6863;0.2471 0.1608 0.698;0.251 0.1647 0.7059;0.251 0.1686 0.7176;0.2549 0.1725 0.7294;0.2549 0.1765 0.7412;0.2588 0.1804 0.749;0.2588 0.1843 0.7608;0.2627 0.1922 0.7843;0.2627 0.1961 0.7922;0.2667 0.2 0.8039;0.2667 0.2039 0.8157;0.2706 0.2078 0.8235;0.2706 0.2157 0.8353;0.2706 0.2196 0.8431;0.2745 0.2235 0.851;0.2745 0.2275 0.8627;0.2745 0.2314 0.8706;0.2745 0.2392 0.8784;0.2784 0.2431 0.8824;0.2784 0.2471 0.8902;0.2784 0.2549 0.898;0.2784 0.2588 0.902;0.2784 0.2667 0.9098;0.2784 0.2706 0.9137;0.2784 0.2745 0.9216;0.2824 0.2824 0.9255;0.2824 0.2863 0.9294;0.2824 0.2941 0.9333;0.2824 0.298 0.9412;0.2824 0.3059 0.9451;0.2824 0.3098 0.949;0.2824 0.3137 0.9529;0.2824 0.3216 0.9569;0.2824 0.3255 0.9608;0.2824 0.3294 0.9647;0.2784 0.3373 0.9686;0.2784 0.3412 0.9686;0.2784 0.349 0.9725;0.2784 0.3529 0.9765;0.2784 0.3569 0.9804;0.2784 0.3647 0.9804;0.2745 0.3686 0.9843;0.2745 0.3765 0.9843;0.2745 0.3804 0.9882;0.2706 0.3843 0.9882;0.2706 0.3922 0.9922;0.2667 0.3961 0.9922;0.2627 0.4039 0.9922;0.2627 0.4078 0.9961;0.2588 0.4157 0.9961;0.2549 0.4196 0.9961;0.251 0.4275 0.9961;0.2471 0.4314 1;0.2431 0.4392 1;0.2353 0.4431 1;0.2314 0.451 1;0.2235 0.4549 1;0.2196 0.4627 0.9961;0.2118 0.4667 0.9961;0.2078 0.4745 0.9922;0.2 0.4784 0.9922;0.1961 0.4863 0.9882;0.1922 0.4902 0.9882;0.1882 0.498 0.9843;0.1843 0.502 0.9804;0.1843 0.5098 0.9804;0.1804 0.5137 0.9765;0.1804 0.5176 0.9725;0.1804 0.5255 0.9725;0.1804 0.5294 0.9686;0.1765 0.5333 0.9647;0.1765 0.5412 0.9608;0.1765 0.5451 0.9569;0.1765 0.549 0.9529;0.1765 0.5569 0.949;0.1725 0.5608 0.9451;0.1725 0.5647 0.9412;0.1686 0.5686 0.9373;0.1647 0.5765 0.9333;0.1608 0.5804 0.9294;0.1569 0.5843 0.9255;0.1529 0.5922 0.9216;0.1529 0.5961 0.9176;0.149 0.6 0.9137;0.149 0.6039 0.9098;0.1451 0.6078 0.9098;0.1451 0.6118 0.9059;0.1412 0.6196 0.902;0.1412 0.6235 0.898;0.1373 0.6275 0.898;0.1373 0.6314 0.8941;0.1333 0.6353 0.8941;0.1294 0.6392 0.8902;0.1255 0.6471 0.8902;0.1216 0.651 0.8863;0.1176 0.6549 0.8824;0.1137 0.6588 0.8824;0.1137 0.6627 0.8784;0.1098 0.6667 0.8745;0.1059 0.6706 0.8706;0.102 0.6745 0.8667;0.098 0.6784 0.8627;0.0902 0.6824 0.8549;0.0863 0.6863 0.851;0.0784 0.6902 0.8471;0.0706 0.6941 0.8392;0.0627 0.698 0.8353;0.0549 0.702 0.8314;0.0431 0.702 0.8235;0.0314 0.7059 0.8196;0.0235 0.7098 0.8118;0.0157 0.7137 0.8078;0.0078 0.7176 0.8;0.0039 0.7176 0.7922;0 0.7216 0.7882;0 0.7255 0.7804;0 0.7294 0.7765;0.0039 0.7294 0.7686;0.0078 0.7333 0.7608;0.0157 0.7333 0.7569;0.0235 0.7373 0.749;0.0353 0.7412 0.7412;0.051 0.7412 0.7373;0.0627 0.7451 0.7294;0.0784 0.7451 0.7216;0.0902 0.749 0.7137;0.102 0.7529 0.7098;0.1137 0.7529 0.702;0.1255 0.7569 0.6941;0.1373 0.7569 0.6863;0.1451 0.7608 0.6824;0.1529 0.7608 0.6745;0.1608 0.7647 0.6667;0.1686 0.7647 0.6588;0.1725 0.7686 0.651;0.1804 0.7686 0.6471;0.1843 0.7725 0.6392;0.1922 0.7725 0.6314;0.1961 0.7765 0.6235;0.2 0.7804 0.6157;0.2078 0.7804 0.6078;0.2118 0.7843 0.6;0.2196 0.7843 0.5882;0.2235 0.7882 0.5804;0.2314 0.7882 0.5725;0.2392 0.7922 0.5647;0.251 0.7922 0.5529;0.2588 0.7922 0.5451;0.2706 0.7961 0.5373;0.2824 0.7961 0.5255;0.2941 0.7961 0.5176;0.3059 0.8 0.5059;0.3176 0.8 0.498;0.3294 0.8 0.4863;0.3412 0.8 0.4784;0.3529 0.8 0.4667;0.3686 0.8039 0.4549;0.3804 0.8039 0.4471;0.3922 0.8039 0.4353;0.4039 0.8039 0.4235;0.4196 0.8039 0.4118;0.4314 0.8039 0.4;0.4471 0.8039 0.3922;0.4627 0.8 0.3804;0.4745 0.8 0.3686;0.4902 0.8 0.3569;0.5059 0.8 0.349;0.5176 0.8 0.3373;0.5333 0.7961 0.3255;0.5451 0.7961 0.3176;0.5608 0.7961 0.3059;0.5765 0.7922 0.2941;0.5882 0.7922 0.2824;0.6039 0.7882 0.2745;0.6157 0.7882 0.2627;0.6314 0.7843 0.251;0.6431 0.7843 0.2431;0.6549 0.7804 0.2314;0.6706 0.7804 0.2235;0.6824 0.7765 0.2157;0.698 0.7765 0.2078;0.7098 0.7725 0.2;0.7216 0.7686 0.1922;0.7333 0.7686 0.1843;0.7451 0.7647 0.1765;0.7608 0.7647 0.1725;0.7725 0.7608 0.1647;0.7843 0.7569 0.1608;0.7961 0.7569 0.1569;0.8078 0.7529 0.1529;0.8157 0.749 0.1529;0.8275 0.749 0.1529;0.8392 0.7451 0.1529;0.851 0.7451 0.1569;0.8588 0.7412 0.1569;0.8706 0.7373 0.1608;0.8824 0.7373 0.1647;0.8902 0.7373 0.1686;0.902 0.7333 0.1765;0.9098 0.7333 0.1804;0.9176 0.7294 0.1882;0.9255 0.7294 0.1961;0.9373 0.7294 0.2078;0.9451 0.7294 0.2157;0.9529 0.7294 0.2235;0.9608 0.7294 0.2314;0.9686 0.7294 0.2392;0.9765 0.7294 0.2431;0.9843 0.7333 0.2431;0.9882 0.7373 0.2431;0.9961 0.7412 0.2392;0.9961 0.7451 0.2353;0.9961 0.7529 0.2314;0.9961 0.7569 0.2275;0.9961 0.7608 0.2235;0.9961 0.7686 0.2196;0.9961 0.7725 0.2157;0.9961 0.7804 0.2078;0.9961 0.7843 0.2039;0.9961 0.7922 0.2;0.9922 0.7961 0.1961;0.9922 0.8039 0.1922;0.9922 0.8078 0.1922;0.9882 0.8157 0.1882;0.9843 0.8235 0.1843;0.9843 0.8275 0.1804;0.9804 0.8353 0.1804;0.9765 0.8392 0.1765;0.9765 0.8471 0.1725;0.9725 0.851 0.1686;0.9686 0.8588 0.1647;0.9686 0.8667 0.1647;0.9647 0.8706 0.1608;0.9647 0.8784 0.1569;0.9608 0.8824 0.1569;0.9608 0.8902 0.1529;0.9608 0.898 0.149;0.9608 0.902 0.149;0.9608 0.9098 0.1451;0.9608 0.9137 0.1412;0.9608 0.9216 0.1373;0.9608 0.9255 0.1333;0.9608 0.9333 0.1294;0.9647 0.9373 0.1255;0.9647 0.9451 0.1216;0.9647 0.949 0.1176;0.9686 0.9569 0.1098;0.9686 0.9608 0.1059;0.9725 0.9686 0.102;0.9725 0.9725 0.0941;0.9765 0.9765 0.0863;0.9765 0.9843 0.0824];
                app.effimg.FontSize = 13;
                app.effimg.NextPlot = 'replace';
                app.effimg.Tag = 'effimg';
                app.effimg.Position = [0 30 320 320];
                
                % Create Label_QUALITY
                app.Label_QUALITY = uilabel(app.figure);
                app.Label_QUALITY.FontWeight = 'bold';
                app.Label_QUALITY.FontColor = [0.118 0.565 1]; % 深海蓝色
                app.Label_QUALITY.Position = [800 270 400 200];
                app.Label_QUALITY.FontName = '宋体';
                app.Label_QUALITY.FontSize = 27;
                app.Label_QUALITY.Text = '去噪效果';
                % Create Label_MSE
                app.Label_MSE = uilabel(app.figure);
                app.Label_MSE.FontWeight = 'bold';
                app.Label_MSE.FontColor = [0.118 0.565 1]; % 深海蓝色
                app.Label_MSE.Position = [780 230 400 200];
                app.Label_MSE.FontName = '宋体';
                app.Label_MSE.FontSize = 20;
                app.Label_MSE.Text = 'MSE ：';
                % Create Label_SNR
                app.Label_SNR = uilabel(app.figure);
                app.Label_SNR.FontWeight = 'bold';
                app.Label_SNR.FontColor = [0.118 0.565 1]; % 深海蓝色
                app.Label_SNR.Position = [780 200 400 200];
                app.Label_SNR.FontName = '宋体';
                app.Label_SNR.FontSize = 20;
                app.Label_SNR.Text = 'SNR ：';
                % Create Label_PSNR
                app.Label_PSNR = uilabel(app.figure);
                app.Label_PSNR.FontWeight = 'bold';
                app.Label_PSNR.FontColor = [0.118 0.565 1]; % 深海蓝色
                app.Label_PSNR.Position = [780 170 400 200];
                app.Label_PSNR.FontName = '宋体';
                app.Label_PSNR.FontSize = 20;
                app.Label_PSNR.Text = 'PSNR： ';
                % Create Label_SSIM
                app.Label_SSIM = uilabel(app.figure);
                app.Label_SSIM.FontWeight = 'bold';
                app.Label_SSIM.FontColor = [0.118 0.565 1]; % 深海蓝色
                app.Label_SSIM.Position = [780 140 400 200];
                app.Label_SSIM.FontName = '宋体';
                app.Label_SSIM.FontSize = 20;
                app.Label_SSIM.Text = 'SSIM：';
                % Create Label_qualityINFO
                app.Label_qualityINFO = uilabel(app.figure);
                app.Label_qualityINFO.FontColor = [0.139 0.79 0.349]; % 绿色
                app.Label_qualityINFO.Position = [760 19 400 200];
                %                 app.Label_qualityINFO.FontWeight = 'bold';
                app.Label_qualityINFO.FontSize = 13;
                app.Label_qualityINFO.Text = [
                    'MSE（[0, +∞)）：' , ...
                    newline, ...
                    '  数值越小表示图像质量更好;',...
                    newline, ...
                    'SNR（[0, +∞) dB）：', ...
                    newline, ...
                    '  数值越高表示图像信号与噪声', ...
                    newline, ...
                    '  比率更高,图像质量更好；',...
                    newline, ...
                    'PSNR（[0, +∞) dB）：', ...
                    newline, ...
                    '  数值越高表示图像质量更好；', ...
                    newline, ...
                    'SSIM（[-1, 1]）：', ...
                    newline, ...
                    '  接近 1 表示图像质量更好；'
                    ];
                % Create Label
                app.Label_Wn = uilabel(app.figure);
                app.Label_Wn.FontWeight = 'bold';
                app.Label_Wn.FontSize = 13;
                app.Label_Wn.FontColor = [0.139 0.79 0.349];
                app.Label_Wn.Position = [775 5  200 35];
                app.Label_Wn.Text = {'   注：  维纳滤波需要'; '在添加运动噪波之后去噪'; ''};
                
            catch
                errordlg('发生了错误，请检查输入或操作','错误','modal');
            end
            % Create uibuttongroup8 频域滤波
            try
                app.uibuttongroup8 = uibuttongroup(app.figure);
                app.uibuttongroup8.Title = '频域滤波';
                app.uibuttongroup8.Tag = 'uibuttongroup8';
                app.uibuttongroup8.FontName = 'Microsoft YaHei UI';
                app.uibuttongroup8.FontSize = 16;
                app.uibuttongroup8.Position = [185 5 390 380];
                % Create p21
                app.p21 = uibutton(app.uibuttongroup8, 'push');
                app.p21.ButtonPushedFcn = createCallbackFcn(app, @p21_Callback, true);
                app.p21.Tag = 'p21';
                app.p21.FontSize = 16;
                app.p21.Position = [15 270 105 28];
                app.p21.Text = '理想低通滤波';
                % Create Label_idealp_Fre
                app.Label_idealp_Fre = uilabel(app.uibuttongroup8);
                app.Label_idealp_Fre.FontSize = 14;
                app.Label_idealp_Fre.FontWeight = 'bold';
                app.Label_idealp_Fre.FontColor = [1 0 0];
                app.Label_idealp_Fre.Position = [13 290 80 50];
                app.Label_idealp_Fre.Text = {'截止频率:'};
                % Create EditField_idealp_Fre
                app.EditField_idealp_Fre = uieditfield(app.uibuttongroup8, 'numeric');
                app.EditField_idealp_Fre.ValueChangedFcn = createCallbackFcn(app, @EditField_idealp_FreChanged, true);
                app.EditField_idealp_Fre.Position = [75 305 40 19];
                app.EditField_idealp_Fre.Value =60;
                % Create p22
                app.p22 = uibutton(app.uibuttongroup8, 'push');
                app.p22.ButtonPushedFcn = createCallbackFcn(app, @p22_Callback, true);
                app.p22.Tag = 'p22';
                app.p22.FontSize = 16;
                app.p22.Position = [15 220 105 28];
                app.p22.Text = '理想高通滤波';
                % Create Label_idealp_Fre
                app.Label_ideahp_Fre = uilabel(app.uibuttongroup8);
                app.Label_ideahp_Fre.FontSize = 14;
                app.Label_ideahp_Fre.FontWeight = 'bold';
                app.Label_ideahp_Fre.FontColor = [1 0 0];
                app.Label_ideahp_Fre.Position = [13 235 80 50];
                app.Label_ideahp_Fre.Text = {'截止频率:'};
                % Create EditField_idealp_Fre
                app.EditField_ideahp_Fre = uieditfield(app.uibuttongroup8, 'numeric');
                app.EditField_ideahp_Fre.ValueChangedFcn = createCallbackFcn(app, @EditField_ideahp_FreChanged, true);
                app.EditField_ideahp_Fre.Position = [75 250 40 19];
                app.EditField_ideahp_Fre.Value =30;
                % Create p23
                app.p23 = uibutton(app.uibuttongroup8, 'push');
                app.p23.ButtonPushedFcn = createCallbackFcn(app, @p23_Callback, true);
                app.p23.Tag = 'p23';
                app.p23.FontSize = 16;
                app.p23.Position = [15 150 105 28];
                app.p23.Text = '理想带通滤波';
                % Create Label_ideabp_lFre
                app.Label_ideabp_lFre = uilabel(app.uibuttongroup8);
                app.Label_ideabp_lFre.FontSize = 12;
                app.Label_ideabp_lFre.FontWeight = 'bold';
                app.Label_ideabp_lFre.FontColor = [1 0 0];
                app.Label_ideabp_lFre.Position = [3 194 100 30];
                app.Label_ideabp_lFre.Text = {'下限截止频率:'};
                % Create EditField_ideabp_lFre
                app.EditField_ideabp_lFre = uieditfield(app.uibuttongroup8, 'numeric');
                app.EditField_ideabp_lFre.ValueChangedFcn = createCallbackFcn(app, @EditField_ideabp_lFreChanged, true);
                app.EditField_ideabp_lFre.Position = [85 200 40 19];
                app.EditField_ideabp_lFre.Value =20;
                % Create Label_ideabp_hFre
                app.Label_ideabp_hFre = uilabel(app.uibuttongroup8);
                app.Label_ideabp_hFre.FontSize = 12;
                app.Label_ideabp_hFre.FontWeight = 'bold';
                app.Label_ideabp_hFre.FontColor = [1 0 0];
                app.Label_ideabp_hFre.Position = [3 174 100 30];
                app.Label_ideabp_hFre.Text = {'上限截止频率:'};
                % Create EditField_ideabp_hFre
                app.EditField_ideabp_hFre = uieditfield(app.uibuttongroup8, 'numeric');
                app.EditField_ideabp_hFre.ValueChangedFcn = createCallbackFcn(app, @EditField_ideabp_hFreChanged, true);
                app.EditField_ideabp_hFre.Position = [85 180 40 19];
                app.EditField_ideabp_hFre.Value =80;
                % Create p24
                app.p24 = uibutton(app.uibuttongroup8, 'push');
                app.p24.ButtonPushedFcn = createCallbackFcn(app, @p24_Callback, true);
                app.p24.Tag = 'p24';
                app.p24.FontSize = 16;
                app.p24.Position = [15 80 105 28];
                app.p24.Text = '理想带阻滤波';
                % Create Label_ideabs_Fre
                app.Label_ideabs_Fre = uilabel(app.uibuttongroup8);
                app.Label_ideabs_Fre.FontSize = 12;
                app.Label_ideabs_Fre.FontWeight = 'bold';
                app.Label_ideabs_Fre.FontColor = [1 0 0];
                app.Label_ideabs_Fre.Position = [13 116 80 50];
                app.Label_ideabs_Fre.Text = {'截止频率:'};
                % Create EditField_ideabs_Fre
                app.EditField_ideabs_Fre = uieditfield(app.uibuttongroup8, 'numeric');
                app.EditField_ideabs_Fre.ValueChangedFcn = createCallbackFcn(app, @EditField_ideabs_FreChanged, true);
                app.EditField_ideabs_Fre.Position = [75 130 40 19];
                app.EditField_ideabs_Fre.Value =50;
                % Create Label_ideabs_w
                app.Label_ideabs_w = uilabel(app.uibuttongroup8);
                app.Label_ideabs_w.FontSize = 12;
                app.Label_ideabs_w.FontWeight = 'bold';
                app.Label_ideabs_w.FontColor = [1 0 0];
                app.Label_ideabs_w.Position = [13 95 80 50];
                app.Label_ideabs_w.Text = {'阻带带宽:'};
                % Create EditField_ideabs_w
                app.EditField_ideabs_w = uieditfield(app.uibuttongroup8, 'numeric');
                app.EditField_ideabs_w.ValueChangedFcn = createCallbackFcn(app, @EditField_ideabs_wChanged, true);
                app.EditField_ideabs_w.Position = [75 110 40 19];
                app.EditField_ideabs_w.Value =30;
                
                %巴特沃斯滤波器
                %%
                % Create Label_butterOrder
                app.Label_butterOrder = uilabel(app.uibuttongroup8);
                app.Label_butterOrder.FontSize = 12;
                app.Label_butterOrder.FontWeight = 'bold';
                app.Label_butterOrder.FontColor = [1 0 0];
                app.Label_butterOrder.Position = [130 290 80 50];
                app.Label_butterOrder.Text = {'滤波器阶数:'};
                % Create EditField_butterOrder
                app.EditField_butterOrder = uieditfield(app.uibuttongroup8, 'numeric');
                app.EditField_butterOrder.ValueChangedFcn = createCallbackFcn(app, @EditField_butterOrderChanged, true);
                app.EditField_butterOrder.Position = [200 305 40 19];
                app.EditField_butterOrder.Value =6;
                % Create p31
                app.p31 = uibutton(app.uibuttongroup8, 'push');
                app.p31.ButtonPushedFcn = createCallbackFcn(app, @p31_Callback, true);
                app.p31.Tag = 'p31';
                app.p31.FontSize = 16;
                app.p31.Position = [135 253 105 28];
                app.p31.Text = '巴特沃斯低通滤波';
                % Create Label_butterlp_Fre
                app.Label_butterlp_Fre = uilabel(app.uibuttongroup8);
                app.Label_butterlp_Fre.FontSize = 14;
                app.Label_butterlp_Fre.FontWeight = 'bold';
                app.Label_butterlp_Fre.FontColor = [1 0 0];
                app.Label_butterlp_Fre.Position = [130 270 80 50];
                app.Label_butterlp_Fre.Text = {'截止频率:'};
                % Create EditField_butterlp_Fre
                app.EditField_butterlp_Fre = uieditfield(app.uibuttongroup8, 'numeric');
                app.EditField_butterlp_Fre.ValueChangedFcn = createCallbackFcn(app, @EditField_butterlp_FreChanged, true);
                app.EditField_butterlp_Fre.Position = [200 285 40 19];
                app.EditField_butterlp_Fre.Value =60;
                % Create p32
                app.p32 = uibutton(app.uibuttongroup8, 'push');
                app.p32.ButtonPushedFcn = createCallbackFcn(app, @p32_Callback, true);
                app.p32.Tag = 'p32';
                app.p32.FontSize = 16;
                app.p32.Position = [135 200 105 28];
                app.p32.Text = '巴特沃斯高通滤波';
                % Create Label_butterhp_Fre
                app.Label_butterhp_Fre = uilabel(app.uibuttongroup8);
                app.Label_butterhp_Fre.FontSize = 14;
                app.Label_butterhp_Fre.FontWeight = 'bold';
                app.Label_butterhp_Fre.FontColor = [1 0 0];
                app.Label_butterhp_Fre.Position = [130 215 80 50];
                app.Label_butterhp_Fre.Text = {'截止频率:'};
                % Create EditField_butterhp_Fre
                app.EditField_butterhp_Fre = uieditfield(app.uibuttongroup8, 'numeric');
                app.EditField_butterhp_Fre.ValueChangedFcn = createCallbackFcn(app, @EditField_butterhp_FreChanged, true);
                app.EditField_butterhp_Fre.Position = [200 230 40 19];
                app.EditField_butterhp_Fre.Value =30;
                % Create p33
                app.p33 = uibutton(app.uibuttongroup8, 'push');
                app.p33.ButtonPushedFcn = createCallbackFcn(app, @p33_Callback, true);
                app.p33.Tag = 'p33';
                app.p33.FontSize = 16;
                app.p33.Position = [135 130 105 28];
                app.p33.Text = '巴特沃斯带通滤波';
                % Create Label_butterbp_lFre
                app.Label_butterbp_lFre = uilabel(app.uibuttongroup8);
                app.Label_butterbp_lFre.FontSize = 12;
                app.Label_butterbp_lFre.FontWeight = 'bold';
                app.Label_butterbp_lFre.FontColor = [1 0 0];
                app.Label_butterbp_lFre.Position = [125 163 80 50];
                app.Label_butterbp_lFre.Text = {'下限截止频率:'};
                % Create EditField_butterbp_lFre
                app.EditField_butterbp_lFre = uieditfield(app.uibuttongroup8, 'numeric');
                app.EditField_butterbp_lFre.ValueChangedFcn = createCallbackFcn(app, @EditField_butterbp_lFreChanged, true);
                app.EditField_butterbp_lFre.Position = [205 179 40 19];
                app.EditField_butterbp_lFre.Value =20;
                % Create Label_butterbp_hFre
                app.Label_butterbp_hFre = uilabel(app.uibuttongroup8);
                app.Label_butterbp_hFre.FontSize = 12;
                app.Label_butterbp_hFre.FontWeight = 'bold';
                app.Label_butterbp_hFre.FontColor = [1 0 0];
                app.Label_butterbp_hFre.Position = [125 143 80 50];
                app.Label_butterbp_hFre.Text = {'上限截止频率:'};
                % Create EditField_butterbp_hFre
                app.EditField_butterbp_hFre = uieditfield(app.uibuttongroup8, 'numeric');
                app.EditField_butterbp_hFre.ValueChangedFcn = createCallbackFcn(app, @EditField_butterbp_hFreChanged, true);
                app.EditField_butterbp_hFre.Position = [205 159 40 19];
                app.EditField_butterbp_hFre.Value =80;
                % Create p34
                app.p34 = uibutton(app.uibuttongroup8, 'push');
                app.p34.ButtonPushedFcn = createCallbackFcn(app, @p34_Callback, true);
                app.p34.Tag = 'p34';
                app.p34.FontSize = 16;
                app.p34.Position = [135 60 105 28];
                app.p34.Text = '巴特沃斯带阻滤波';
                % Create Label_butterbs_Fre
                app.Label_butterbs_Fre = uilabel(app.uibuttongroup8);
                app.Label_butterbs_Fre.FontSize = 12;
                app.Label_butterbs_Fre.FontWeight = 'bold';
                app.Label_butterbs_Fre.FontColor = [1 0 0];
                app.Label_butterbs_Fre.Position = [135 95 80 50];
                app.Label_butterbs_Fre.Text = {'截止频率:'};
                % Create EditField_butterbs_Fre
                app.EditField_butterbs_Fre = uieditfield(app.uibuttongroup8, 'numeric');
                app.EditField_butterbs_Fre.ValueChangedFcn = createCallbackFcn(app, @EditField_butterbs_FreChanged, true);
                app.EditField_butterbs_Fre.Position = [200 110 40 19];
                app.EditField_butterbs_Fre.Value =50;
                % Create Label_butterbs_w
                app.Label_butterbs_w = uilabel(app.uibuttongroup8);
                app.Label_butterbs_w.FontSize = 12;
                app.Label_butterbs_w.FontWeight = 'bold';
                app.Label_butterbs_w.FontColor = [1 0 0];
                app.Label_butterbs_w.Position = [135 74 80 50];
                app.Label_butterbs_w.Text = {'阻带带宽:'};
                % Create EditField_butterbs_w
                app.EditField_butterbs_w = uieditfield(app.uibuttongroup8, 'numeric');
                app.EditField_butterbs_w.ValueChangedFcn = createCallbackFcn(app, @EditField_butterbs_wChanged, true);
                app.EditField_butterbs_w.Position = [200 90 40 19];
                app.EditField_butterbs_w.Value =30;
                %%
                %高斯滤波器
                % Create f21
                app.f21 = uibutton(app.uibuttongroup8, 'push');
                app.f21.ButtonPushedFcn = createCallbackFcn(app, @f21_Callback, true);
                app.f21.Tag = 'f21';
                app.f21.FontSize = 16;
                app.f21.Position = [270 270 105 28];
                app.f21.Text = '高斯低通滤波';
                % Create Label_gausslp_Fre
                app.Label_gausslp_Fre = uilabel(app.uibuttongroup8);
                app.Label_gausslp_Fre.FontSize = 14;
                app.Label_gausslp_Fre.FontWeight = 'bold';
                app.Label_gausslp_Fre.FontColor = [1 0 0];
                app.Label_gausslp_Fre.Position = [260 290 80 50];
                app.Label_gausslp_Fre.Text = {'截止频率:'};
                % Create EditField_gausslp_Fre
                app.EditField_gausslp_Fre = uieditfield(app.uibuttongroup8, 'numeric');
                app.EditField_gausslp_Fre.ValueChangedFcn = createCallbackFcn(app, @EditField_gausslp_FreChanged, true);
                app.EditField_gausslp_Fre.Position = [330 305 40 19];
                app.EditField_gausslp_Fre.Value =60;
                % Create f22
                app.f22 = uibutton(app.uibuttongroup8, 'push');
                app.f22.ButtonPushedFcn = createCallbackFcn(app, @f22_Callback, true);
                app.f22.Tag = 'f22';
                app.f22.FontSize = 16;
                app.f22.Position = [270 210 105 28];
                app.f22.Text = '高斯高通滤波';
                % Create Label_gausshp_Fre
                app.Label_gausshp_Fre = uilabel(app.uibuttongroup8);
                app.Label_gausshp_Fre.FontSize = 14;
                app.Label_gausshp_Fre.FontWeight = 'bold';
                app.Label_gausshp_Fre.FontColor = [1 0 0];
                app.Label_gausshp_Fre.Position = [260 240 80 30];
                app.Label_gausshp_Fre.Text = {'截止频率:'};
                % Create EditField_gausshp_Fre
                app.EditField_gausshp_Fre = uieditfield(app.uibuttongroup8, 'numeric');
                app.EditField_gausshp_Fre.ValueChangedFcn = createCallbackFcn(app, @EditField_gausshp_FreChanged, true);
                app.EditField_gausshp_Fre.Position = [330 245 40 19];
                app.EditField_gausshp_Fre.Value =30;
                % Create f23
                app.f23 = uibutton(app.uibuttongroup8, 'push');
                app.f23.ButtonPushedFcn = createCallbackFcn(app, @f23_Callback, true);
                app.f23.Tag = 'f23';
                app.f23.FontSize = 16;
                app.f23.Position = [270 130 105 28];
                app.f23.Text = '高斯带通滤波';
                % Create Label_gaussbp_lFre
                app.Label_gaussbp_lFre = uilabel(app.uibuttongroup8);
                app.Label_gaussbp_lFre.FontSize = 12;
                app.Label_gaussbp_lFre.FontWeight = 'bold';
                app.Label_gaussbp_lFre.FontColor = [1 0 0];
                app.Label_gaussbp_lFre.Position = [260 185 80 30];
                app.Label_gaussbp_lFre.Text = {'下限截止频率:'};
                % Create EditField_gaussbp_lFre
                app.EditField_gaussbp_lFre = uieditfield(app.uibuttongroup8, 'numeric');
                app.EditField_gaussbp_lFre.ValueChangedFcn = createCallbackFcn(app, @EditField_gaussbp_lFreChanged, true);
                app.EditField_gaussbp_lFre.Position = [340 190 40 19];
                app.EditField_gaussbp_lFre.Value =20;
                % Create Label_gaussbp_hFre
                app.Label_gaussbp_hFre = uilabel(app.uibuttongroup8);
                app.Label_gaussbp_hFre.FontSize = 12;
                app.Label_gaussbp_hFre.FontWeight = 'bold';
                app.Label_gaussbp_hFre.FontColor = [1 0 0];
                app.Label_gaussbp_hFre.Position = [260 160 80 30];
                app.Label_gaussbp_hFre.Text = {'上限截止频率:'};
                % Create EditField_gaussbp_hFre
                app.EditField_gaussbp_hFre = uieditfield(app.uibuttongroup8, 'numeric');
                app.EditField_gaussbp_hFre.ValueChangedFcn = createCallbackFcn(app, @EditField_gaussbp_hFreChanged, true);
                app.EditField_gaussbp_hFre.Position = [340 165 40 19];
                app.EditField_gaussbp_hFre.Value =80;
                % Create f24
                app.f24 = uibutton(app.uibuttongroup8, 'push');
                app.f24.ButtonPushedFcn = createCallbackFcn(app, @f24_Callback, true);
                app.f24.Tag = 'f24';
                app.f24.FontSize = 16;
                app.f24.Position = [270 60 105 28];
                app.f24.Text = '高斯带阻滤波';
                % Create Label_gaussbs_Fre
                app.Label_gaussbs_Fre = uilabel(app.uibuttongroup8);
                app.Label_gaussbs_Fre.FontSize = 12;
                app.Label_gaussbs_Fre.FontWeight = 'bold';
                app.Label_gaussbs_Fre.FontColor = [1 0 0];
                app.Label_gaussbs_Fre.Position = [265 105 80 30];
                app.Label_gaussbs_Fre.Text = {'截止频率:'};
                % Create EditField_gaussbs_Fre
                app.EditField_gaussbs_Fre = uieditfield(app.uibuttongroup8, 'numeric');
                app.EditField_gaussbs_Fre.ValueChangedFcn = createCallbackFcn(app, @EditField_gaussbs_FreChanged, true);
                app.EditField_gaussbs_Fre.Position = [330 110 40 19];
                app.EditField_gaussbs_Fre.Value =50;
                % Create Label_gaussbs_w
                app.Label_gaussbs_w = uilabel(app.uibuttongroup8);
                app.Label_gaussbs_w.FontSize = 12;
                app.Label_gaussbs_w.FontWeight = 'bold';
                app.Label_gaussbs_w.FontColor = [1 0 0];
                app.Label_gaussbs_w.Position = [265 84 80 30];
                app.Label_gaussbs_w.Text = {'阻带带宽:'};
                % Create EditField_gaussbs_w
                app.EditField_gaussbs_w = uieditfield(app.uibuttongroup8, 'numeric');
                app.EditField_gaussbs_w.ValueChangedFcn = createCallbackFcn(app, @EditField_butterbs_wChanged, true);
                app.EditField_gaussbs_w.Position = [330 90 40 19];
                app.EditField_gaussbs_w.Value =30;
                
                % Create p4
                app.p4 = uibutton(app.uibuttongroup8, 'push');
                app.p4.ButtonPushedFcn = createCallbackFcn(app, @p4_Callback, true);
                app.p4.Tag = 'p4';
                app.p4.FontSize = 16;
                app.p4.Position = [10 10 110 28];
                app.p4.Text = 'haar小波去噪';
                % Create Label_wtThreshold
                app.Label_wtThreshold = uilabel(app.uibuttongroup8);
                app.Label_wtThreshold.FontSize = 13;
                app.Label_wtThreshold.FontWeight = 'bold';
                app.Label_wtThreshold.FontColor = [1 0 0];
                app.Label_wtThreshold.Position = [10 45 90 30];
                app.Label_wtThreshold.Text = {'小波去噪';'  阈值:'};
                % Create EditField_wtThreshold
                app.EditField_wtThreshold = uieditfield(app.uibuttongroup8, 'numeric');
                app.EditField_wtThreshold.ValueChangedFcn = createCallbackFcn(app, @EditField_wtThresholdChanged, true);
                app.EditField_wtThreshold.Position = [75 50 40 19];
                app.EditField_wtThreshold.Value =0.5;
                
                
                app.homefilter = uibuttongroup(app.uibuttongroup8);
                app.homefilter.Position = [128 2 260 55];
                % Create p5
                app.p5 = uibutton(app.homefilter, 'push');
                app.p5.ButtonPushedFcn = createCallbackFcn(app, @p5_Callback, true);
                app.p5.Tag = 'p5';
                app.p5.FontSize = 16;
                app.p5.Position = [135 0 85.3333333333333 28];
                app.p5.Text = '同态滤波';
                % Create Label_homeOrder
                app.Label_homeOrder = uilabel(app.homefilter);
                app.Label_homeOrder.FontSize = 14;
                app.Label_homeOrder.FontWeight = 'bold';
                app.Label_homeOrder.FontColor = [1 0 0];
                app.Label_homeOrder.Position = [5 25 80 30];
                app.Label_homeOrder.Text = {'阶数:'};
                % Create EditField_homeOrder
                app.EditField_homeOrder = uieditfield(app.homefilter, 'numeric');
                app.EditField_homeOrder.ValueChangedFcn = createCallbackFcn(app, @EditField_homeOrderChanged, true);
                app.EditField_homeOrder.Position = [40 30 40 19];
                app.EditField_homeOrder.Value =1;
                % Create Label_Fre
                app.Label_Fre = uilabel(app.homefilter);
                app.Label_Fre.FontSize = 12;
                app.Label_Fre.FontWeight = 'bold';
                app.Label_Fre.FontColor = [1 0 0];
                app.Label_Fre.Position = [5 1 80 25];
                app.Label_Fre.Text = {'截止频率:'};
                % Create EditField_Fre
                app.EditField_Fre = uieditfield(app.homefilter, 'numeric');
                app.EditField_Fre.ValueChangedFcn = createCallbackFcn(app, @EditField_FreChanged, true);
                app.EditField_Fre.Position = [65 4 40 19];
                app.EditField_Fre.Value =5;
                % Create Label_hGain
                app.Label_hGain = uilabel(app.homefilter);
                app.Label_hGain.FontSize = 12;
                app.Label_hGain.FontWeight = 'bold';
                app.Label_hGain.FontColor = [1 0 0];
                app.Label_hGain.Position = [85 25 80 30];
                app.Label_hGain.Text = {'高频增益:'};
                % Create EditField_hGain
                app.EditField_hGain = uieditfield(app.homefilter, 'numeric');
                app.EditField_hGain.ValueChangedFcn = createCallbackFcn(app, @EditField_hGainChanged, true);
                app.EditField_hGain.Position = [140 30 30 19];
                app.EditField_hGain.Value =1.1;
                % Create Label_lGain
                app.Label_lGain = uilabel(app.homefilter);
                app.Label_lGain.FontSize = 12;
                app.Label_lGain.FontWeight = 'bold';
                app.Label_lGain.FontColor = [1 0 0];
                app.Label_lGain.Position = [173 15 80 50];
                app.Label_lGain.Text = {'低频增益:'};
                % Create EditField_lGain
                app.EditField_lGain = uieditfield(app.homefilter, 'numeric');
                app.EditField_lGain.ValueChangedFcn = createCallbackFcn(app, @EditField_lGainChanged, true);
                app.EditField_lGain.Position = [225 30 30 19];
                app.EditField_lGain.Value =0.1;
            catch
                errordlg('发生了错误，请检查输入或操作','错误','modal');
            end
            % Create uibuttongroup9 空域滤波
            try
                
                app.uibuttongroup9 = uibuttongroup(app.figure);
                app.uibuttongroup9.Title = '空间滤波器/去噪';
                app.uibuttongroup9.Tag = 'uibuttongroup2';
                app.uibuttongroup9.FontName = 'Microsoft YaHei UI';
                app.uibuttongroup9.FontSize = 16;
                app.uibuttongroup9.Position = [580 5 158 390];
                
                % Create Label
                app.Label_Avg = uilabel(app.uibuttongroup9);
                app.Label_Avg.FontWeight = 'bold';
                app.Label_Avg.FontSize = 16;
                app.Label_Avg.FontColor = [0.976 0.549 0.223]; % 鲜艳橙色
                app.Label_Avg.Position = [7 335 200 40];
                app.Label_Avg.Text = {'均值滤波方式:'};
                
                % Create f1_1
                app.f1_1 = uibutton(app.uibuttongroup9, 'push');
                app.f1_1.ButtonPushedFcn = createCallbackFcn(app, @f1_1_Callback, true);
                app.f1_1.Tag = 'f1_1';
                app.f1_1.FontName = 'Microsoft JhengHei';
                app.f1_1.FontSize = 13;
                app.f1_1.Position = [30 300 100 25];
                app.f1_1.Text = 'replicate';
                % Create f1_2
                app.f1_2 = uibutton(app.uibuttongroup9, 'push');
                app.f1_2.ButtonPushedFcn = createCallbackFcn(app, @f1_2_Callback, true);
                app.f1_2.Tag = 'f1_2';
                app.f1_2.FontName = 'Microsoft JhengHei';
                app.f1_2.FontSize = 13;
                app.f1_2.Position = [30 275 100 25];
                app.f1_2.Text = 'symmetric';
                % Create f1_3
                app.f1_3 = uibutton(app.uibuttongroup9, 'push');
                app.f1_3.ButtonPushedFcn = createCallbackFcn(app, @f1_3_Callback, true);
                app.f1_3.Tag = 'f1_3';
                app.f1_3.FontName = 'Microsoft JhengHei';
                app.f1_3.FontSize = 13;
                app.f1_3.Position = [30 250 100 25];
                app.f1_3.Text = 'circular';
                % Create Label_MeanFilter
                app.Label_MeanFilter = uilabel(app.uibuttongroup9);
                app.Label_MeanFilter.FontSize = 13;
                app.Label_MeanFilter.FontWeight = 'bold';
                app.Label_MeanFilter.FontColor = [1 0 0];
                app.Label_MeanFilter.Position = [15 325 80 30];
                app.Label_MeanFilter.Text = {'领域大小:'};
                % Create EditField_MeanFilter
                app.EditField_MeanFilter = uieditfield(app.uibuttongroup9, 'numeric');
                app.EditField_MeanFilter.ValueChangedFcn = createCallbackFcn(app, @EditField_MeanFilterChanged, true);
                app.EditField_MeanFilter.Position = [90 328 40 19];
                app.EditField_MeanFilter.Value =6;
                
                % Create f3
                app.f3 = uibutton(app.uibuttongroup9, 'push');
                app.f3.ButtonPushedFcn = createCallbackFcn(app, @f3_Callback, true);
                app.f3.Tag = 'f3';
                app.f3.FontSize = 18;
                app.f3.Position = [30 198 100 30];
                app.f3.Text = '中值滤波';
                % Create Label_MedianFilter
                app.Label_MedianFilter = uilabel(app.uibuttongroup9);
                app.Label_MedianFilter.FontSize = 13;
                app.Label_MedianFilter.FontWeight = 'bold';
                app.Label_MedianFilter.FontColor = [1 0 0];
                app.Label_MedianFilter.Position = [15 225 80 30];
                app.Label_MedianFilter.Text = {'领域大小:'};
                % Create EditField_MedianFilter
                app.EditField_MedianFilter = uieditfield(app.uibuttongroup9, 'numeric');
                app.EditField_MedianFilter.ValueChangedFcn = createCallbackFcn(app, @EditField_MedianFilterChanged, true);
                app.EditField_MedianFilter.Position = [90 230 40 19];
                app.EditField_MedianFilter.Value =3;
                
                % Create f4
                app.f4 = uibutton(app.uibuttongroup9, 'push');
                app.f4.ButtonPushedFcn = createCallbackFcn(app, @f4_Callback, true);
                app.f4.Tag = 'f4';
                app.f4.FontSize = 18;
                app.f4.Position = [10 35 140 30];
                app.f4.Text = '非局部均值去噪';
                % Create Label_nlmeansH
                app.Label_nlmeansH = uilabel(app.uibuttongroup9);
                app.Label_nlmeansH.FontSize = 12;
                app.Label_nlmeansH.FontWeight = 'bold';
                app.Label_nlmeansH.FontColor = [1 0 0];
                app.Label_nlmeansH.Position = [10 90 80 30];
                app.Label_nlmeansH.Text = {'平滑参数:'};
                % Create EditField_nlmeansH
                app.EditField_nlmeansH = uieditfield(app.uibuttongroup9, 'numeric');
                app.EditField_nlmeansH.ValueChangedFcn = createCallbackFcn(app, @EditField_nlmeansHChanged, true);
                app.EditField_nlmeansH.Position = [90 100 40 16];
                app.EditField_nlmeansH.Value =2;
                % Create Label_patchSize
                app.Label_patchSize = uilabel(app.uibuttongroup9);
                app.Label_patchSize.FontSize = 12;
                app.Label_patchSize.FontWeight = 'bold';
                app.Label_patchSize.FontColor = [1 0 0];
                app.Label_patchSize.Position = [10 75 80 30];
                app.Label_patchSize.Text = {'块大小:'};
                % Create EditField_patchSize
                app.EditField_patchSize = uieditfield(app.uibuttongroup9, 'numeric');
                app.EditField_patchSize.ValueChangedFcn = createCallbackFcn(app, @EditField_patchSizeChanged, true);
                app.EditField_patchSize.Position = [90 83 40 16];
                app.EditField_patchSize.Value =5;
                % Create Label_searchWindowSize
                app.Label_searchWindowSize = uilabel(app.uibuttongroup9);
                app.Label_searchWindowSize.FontSize = 12;
                app.Label_searchWindowSize.FontWeight = 'bold';
                app.Label_searchWindowSize.FontColor = [1 0 0];
                app.Label_searchWindowSize.Position = [10 60 80 30];
                app.Label_searchWindowSize.Text = {'搜索窗口大小:'};
                % Create EditField_searchWindowSize
                app.EditField_searchWindowSize = uieditfield(app.uibuttongroup9, 'numeric');
                app.EditField_searchWindowSize.ValueChangedFcn = createCallbackFcn(app, @EditField_searchWindowSizeChanged, true);
                app.EditField_searchWindowSize.Position = [90 66 40 16];
                app.EditField_searchWindowSize.Value =10;
                
                %总变差去噪
                % Create f5
                app.f5 = uibutton(app.uibuttongroup9, 'push');
                app.f5.ButtonPushedFcn = createCallbackFcn(app, @f5_Callback, true);
                app.f5.Tag = 'f5';
                app.f5.FontSize = 18;
                app.f5.Position = [30 117 100 30];
                app.f5.Text = '总变差去噪';
                % Create Label_lambda
                app.Label_lambda = uilabel(app.uibuttongroup9);
                app.Label_lambda.FontSize = 12;
                app.Label_lambda.FontWeight = 'bold';
                app.Label_lambda.FontColor = [1 0 0];
                app.Label_lambda.Position = [10 175 80 30];
                app.Label_lambda.Text = {'正则化参数:'};
                % Create EditField_lambda
                app.EditField_lambda = uieditfield(app.uibuttongroup9, 'numeric');
                app.EditField_lambda.ValueChangedFcn = createCallbackFcn(app, @EditField_lambdaChanged, true);
                app.EditField_lambda.Position = [90 181 40 16];
                app.EditField_lambda.Value =0.1;
                % Create Label_numIterations
                app.Label_numIterations = uilabel(app.uibuttongroup9);
                app.Label_numIterations.FontSize = 12;
                app.Label_numIterations.FontWeight = 'bold';
                app.Label_numIterations.FontColor = [1 0 0];
                app.Label_numIterations.Position = [10 158 80 30];
                app.Label_numIterations.Text = {'迭代次数:'};
                % Create EditField_numIterations
                app.EditField_numIterations = uieditfield(app.uibuttongroup9, 'numeric');
                app.EditField_numIterations.ValueChangedFcn = createCallbackFcn(app, @EditField_numIterationsChanged, true);
                app.EditField_numIterations.Position = [90 164 40 16];
                app.EditField_numIterations.Value =30;
                % Create Label_deltaT
                app.Label_deltaT = uilabel(app.uibuttongroup9);
                app.Label_deltaT.FontSize = 12;
                app.Label_deltaT.FontWeight = 'bold';
                app.Label_deltaT.FontColor = [1 0 0];
                app.Label_deltaT.Position = [10 142 80 30];
                app.Label_deltaT.Text = {'时间步长:'};
                % Create EditField_deltaT
                app.EditField_deltaT = uieditfield(app.uibuttongroup9, 'numeric');
                app.EditField_deltaT.ValueChangedFcn = createCallbackFcn(app, @EditField_deltaTChanged, true);
                app.EditField_deltaT.Position = [90 147 40 16];
                app.EditField_deltaT.Value =0.1;
                
                % Create p1 维纳滤波
                app.p1 = uibutton(app.uibuttongroup9, 'push');
                app.p1.ButtonPushedFcn = createCallbackFcn(app, @p1_Callback, true);
                app.p1.Tag = 'p1';
                app.p1.FontSize = 18;
                app.p1.Position = [30 3 100 30];
                app.p1.Text = '维纳滤波';
                
            catch
                errordlg('发生了错误，请检查输入或操作','错误','modal');
            end
            % Create uibuttongroup10 噪声添加
            try
                app.uibuttongroup10 = uibuttongroup(app.figure);
                app.uibuttongroup10.Title = '添加噪声';
                app.uibuttongroup10.Tag = 'uibuttongroup10';
                app.uibuttongroup10.FontName = 'Microsoft YaHei UI';
                app.uibuttongroup10.FontSize = 16;
                app.uibuttongroup10.Position = [10 5 170 380];
                
                % Create n1
                app.n1 = uibutton(app.uibuttongroup10, 'push');
                app.n1.ButtonPushedFcn = createCallbackFcn(app, @n1_Callback, true);
                app.n1.Tag = 'n1';
                app.n1.FontSize = 18;
                app.n1.Position = [30,268,100,28];
                app.n1.Text = '高斯白噪声';
                
                % Create Label_MEAN
                app.Label_MEAN = uilabel(app.uibuttongroup10);
                app.Label_MEAN.FontSize = 14;
                app.Label_MEAN.FontWeight = 'bold';
                app.Label_MEAN.FontColor = [1 0 0];
                app.Label_MEAN.Position = [4,300,80,30];
                app.Label_MEAN.Text = {'均值:'};
                
                % Create Label_VAR
                app.Label_VAR = uilabel(app.uibuttongroup10);
                app.Label_VAR.FontSize = 14;
                app.Label_VAR.FontWeight = 'bold';
                app.Label_VAR.FontColor = [1 0 0];
                app.Label_VAR.Position = [88 300 80 30];
                app.Label_VAR.Text = {'方差:'};
                
                % Create EditField_MEAN
                app.EditField_MEAN = uieditfield(app.uibuttongroup10, 'numeric');
                app.EditField_MEAN.ValueChangedFcn = createCallbackFcn(app, @EditField_MEANChanged, true);
                app.EditField_MEAN.Position = [40, 305,40,19];
                app.EditField_MEAN.Value =0;
                
                % Create EditField_VAR
                app.EditField_VAR = uieditfield(app.uibuttongroup10, 'numeric');
                app.EditField_VAR.ValueChangedFcn = createCallbackFcn(app, @EditField_VARChanged, true);
                app.EditField_VAR.Position = [124, 305,40,19];
                app.EditField_VAR.Value = 0.05;
                
                % Create n2
                app.n2 = uibutton(app.uibuttongroup10, 'push');
                app.n2.ButtonPushedFcn = createCallbackFcn(app, @n2_Callback, true);
                app.n2.Tag = 'n2';
                app.n2.FontSize = 18;
                app.n2.Position = [40,230,85,28];
                app.n2.Text = '泊松噪波';
                
                % Create n3
                app.n3 = uibutton(app.uibuttongroup10, 'push');
                app.n3.ButtonPushedFcn = createCallbackFcn(app, @n3_Callback, true);
                app.n3.Tag = 'n3';
                app.n3.FontSize = 18;
                app.n3.Position = [40,170,85,28];
                app.n3.Text = '椒盐噪波';
                
                % Create Label_DENSITY
                app.Label_DENSITY = uilabel(app.uibuttongroup10);
                app.Label_DENSITY.FontSize = 14;
                app.Label_DENSITY.FontWeight = 'bold';
                app.Label_DENSITY.FontColor = [1 0 0];
                app.Label_DENSITY.Position = [8 200 90 30];
                app.Label_DENSITY.Text = {'椒盐噪声密度:'};
                
                % Create EditField_DENSITY
                app.EditField_DENSITY = uieditfield(app.uibuttongroup10, 'numeric');
                app.EditField_DENSITY.ValueChangedFcn = createCallbackFcn(app, @EditField_DENSITYChanged, true);
                app.EditField_DENSITY.Position = [100 206 40 19];
                app.EditField_DENSITY.Value =0.1;
                
                % Create n4
                app.n4 = uibutton(app.uibuttongroup10, 'push');
                app.n4.ButtonPushedFcn = createCallbackFcn(app, @n4_Callback, true);
                app.n4.Tag = 'n4';
                app.n4.FontSize = 18;
                app.n4.Position = [40,116,90,28];
                app.n4.Text = '斑点噪波';
                
                % Create Label_SPEVAR speckle_variance 斑点噪声方差
                app.Label_SPEVAR= uilabel(app.uibuttongroup10);
                app.Label_SPEVAR.FontSize = 14;
                app.Label_SPEVAR.FontWeight = 'bold';
                app.Label_SPEVAR.FontColor = [1 0 0];
                app.Label_SPEVAR.Position = [7 143 99 30];
                app.Label_SPEVAR.Text = {'斑点噪声方差: '};
                
                % Create EditField_SPEVAR
                app.EditField_SPEVAR = uieditfield(app.uibuttongroup10, 'numeric');
                app.EditField_SPEVAR.ValueChangedFcn = createCallbackFcn(app, @EditField_SPEVARChanged, true);
                app.EditField_SPEVAR.Position = [100 148 40 19];
                app.EditField_SPEVAR.Value =0.04;
                
                % Create n5
                app.n5 = uibutton(app.uibuttongroup10, 'push');
                app.n5.ButtonPushedFcn = createCallbackFcn(app, @n5_Callback, true);
                app.n5.Tag = 'n5';
                app.n5.FontSize = 18;
                app.n5.Position = [25, 10,120, 30];
                app.n5.Text = '添加运动噪波';
                
                % Create Label_LEN
                app.Label_LEN = uilabel(app.uibuttongroup10);
                app.Label_LEN.FontSize = 14;
                app.Label_LEN.FontWeight = 'bold';
                app.Label_LEN.FontColor = [1 0 0];
                app.Label_LEN.Position = [18,70,80,50];
                app.Label_LEN.Text = {'线性运动'; '长度:'; ''};
                
                % Create Label_THETA
                app.Label_THETA = uilabel(app.uibuttongroup10);
                app.Label_THETA.FontSize = 14;
                app.Label_THETA.FontWeight = 'bold';
                app.Label_THETA.FontColor = [1 0 0];
                app.Label_THETA.Position = [86 70 80 50];
                app.Label_THETA.Text = {'线性运动'; '方向角度:'};
                
                % Create EditField
                app.EditField_LEN = uieditfield(app.uibuttongroup10, 'numeric');
                app.EditField_LEN.ValueChangedFcn = createCallbackFcn(app, @EditField_LENChanged, true);
                app.EditField_LEN.Position = [18,50,43,19];
                app.EditField_LEN.Value = 20;
                
                % Create EditField
                app.EditField_THETA = uieditfield(app.uibuttongroup10, 'numeric');
                app.EditField_THETA.ValueChangedFcn = createCallbackFcn(app, @EditField_THETAChanged, true);
                app.EditField_THETA.Position = [93,50,43,19];
                app.EditField_THETA.Value = 10;
                
            catch
                errordlg('发生了错误，请检查输入或操作','错误','modal');
            end
            % Create uibuttongroup11 效果图片显示
            try
                app.uibuttongroup11 = uibuttongroup(app.figure);
                app.uibuttongroup11.Title = '效果图片的变换波形显示   (FT,DCT,DWT)';
                app.uibuttongroup11.Tag = 'uibuttongroup11';
                app.uibuttongroup11.FontName = 'Microsoft YaHei UI';
                app.uibuttongroup11.FontSize = 24;
                app.uibuttongroup11.Position = [950 5 660 830];
                
                
                % Create g1 图像RGB颜色分解或者灰度直方图分解
                app.g1 = uiaxes(app.uibuttongroup11);
                app.g1.FontName = 'Microsoft YaHei UI';
                app.g1.Colormap = [0.2431 0.149 0.6588;0.2431 0.1529 0.6745;0.2471 0.1569 0.6863;0.2471 0.1608 0.698;0.251 0.1647 0.7059;0.251 0.1686 0.7176;0.2549 0.1725 0.7294;0.2549 0.1765 0.7412;0.2588 0.1804 0.749;0.2588 0.1843 0.7608;0.2627 0.1922 0.7843;0.2627 0.1961 0.7922;0.2667 0.2 0.8039;0.2667 0.2039 0.8157;0.2706 0.2078 0.8235;0.2706 0.2157 0.8353;0.2706 0.2196 0.8431;0.2745 0.2235 0.851;0.2745 0.2275 0.8627;0.2745 0.2314 0.8706;0.2745 0.2392 0.8784;0.2784 0.2431 0.8824;0.2784 0.2471 0.8902;0.2784 0.2549 0.898;0.2784 0.2588 0.902;0.2784 0.2667 0.9098;0.2784 0.2706 0.9137;0.2784 0.2745 0.9216;0.2824 0.2824 0.9255;0.2824 0.2863 0.9294;0.2824 0.2941 0.9333;0.2824 0.298 0.9412;0.2824 0.3059 0.9451;0.2824 0.3098 0.949;0.2824 0.3137 0.9529;0.2824 0.3216 0.9569;0.2824 0.3255 0.9608;0.2824 0.3294 0.9647;0.2784 0.3373 0.9686;0.2784 0.3412 0.9686;0.2784 0.349 0.9725;0.2784 0.3529 0.9765;0.2784 0.3569 0.9804;0.2784 0.3647 0.9804;0.2745 0.3686 0.9843;0.2745 0.3765 0.9843;0.2745 0.3804 0.9882;0.2706 0.3843 0.9882;0.2706 0.3922 0.9922;0.2667 0.3961 0.9922;0.2627 0.4039 0.9922;0.2627 0.4078 0.9961;0.2588 0.4157 0.9961;0.2549 0.4196 0.9961;0.251 0.4275 0.9961;0.2471 0.4314 1;0.2431 0.4392 1;0.2353 0.4431 1;0.2314 0.451 1;0.2235 0.4549 1;0.2196 0.4627 0.9961;0.2118 0.4667 0.9961;0.2078 0.4745 0.9922;0.2 0.4784 0.9922;0.1961 0.4863 0.9882;0.1922 0.4902 0.9882;0.1882 0.498 0.9843;0.1843 0.502 0.9804;0.1843 0.5098 0.9804;0.1804 0.5137 0.9765;0.1804 0.5176 0.9725;0.1804 0.5255 0.9725;0.1804 0.5294 0.9686;0.1765 0.5333 0.9647;0.1765 0.5412 0.9608;0.1765 0.5451 0.9569;0.1765 0.549 0.9529;0.1765 0.5569 0.949;0.1725 0.5608 0.9451;0.1725 0.5647 0.9412;0.1686 0.5686 0.9373;0.1647 0.5765 0.9333;0.1608 0.5804 0.9294;0.1569 0.5843 0.9255;0.1529 0.5922 0.9216;0.1529 0.5961 0.9176;0.149 0.6 0.9137;0.149 0.6039 0.9098;0.1451 0.6078 0.9098;0.1451 0.6118 0.9059;0.1412 0.6196 0.902;0.1412 0.6235 0.898;0.1373 0.6275 0.898;0.1373 0.6314 0.8941;0.1333 0.6353 0.8941;0.1294 0.6392 0.8902;0.1255 0.6471 0.8902;0.1216 0.651 0.8863;0.1176 0.6549 0.8824;0.1137 0.6588 0.8824;0.1137 0.6627 0.8784;0.1098 0.6667 0.8745;0.1059 0.6706 0.8706;0.102 0.6745 0.8667;0.098 0.6784 0.8627;0.0902 0.6824 0.8549;0.0863 0.6863 0.851;0.0784 0.6902 0.8471;0.0706 0.6941 0.8392;0.0627 0.698 0.8353;0.0549 0.702 0.8314;0.0431 0.702 0.8235;0.0314 0.7059 0.8196;0.0235 0.7098 0.8118;0.0157 0.7137 0.8078;0.0078 0.7176 0.8;0.0039 0.7176 0.7922;0 0.7216 0.7882;0 0.7255 0.7804;0 0.7294 0.7765;0.0039 0.7294 0.7686;0.0078 0.7333 0.7608;0.0157 0.7333 0.7569;0.0235 0.7373 0.749;0.0353 0.7412 0.7412;0.051 0.7412 0.7373;0.0627 0.7451 0.7294;0.0784 0.7451 0.7216;0.0902 0.749 0.7137;0.102 0.7529 0.7098;0.1137 0.7529 0.702;0.1255 0.7569 0.6941;0.1373 0.7569 0.6863;0.1451 0.7608 0.6824;0.1529 0.7608 0.6745;0.1608 0.7647 0.6667;0.1686 0.7647 0.6588;0.1725 0.7686 0.651;0.1804 0.7686 0.6471;0.1843 0.7725 0.6392;0.1922 0.7725 0.6314;0.1961 0.7765 0.6235;0.2 0.7804 0.6157;0.2078 0.7804 0.6078;0.2118 0.7843 0.6;0.2196 0.7843 0.5882;0.2235 0.7882 0.5804;0.2314 0.7882 0.5725;0.2392 0.7922 0.5647;0.251 0.7922 0.5529;0.2588 0.7922 0.5451;0.2706 0.7961 0.5373;0.2824 0.7961 0.5255;0.2941 0.7961 0.5176;0.3059 0.8 0.5059;0.3176 0.8 0.498;0.3294 0.8 0.4863;0.3412 0.8 0.4784;0.3529 0.8 0.4667;0.3686 0.8039 0.4549;0.3804 0.8039 0.4471;0.3922 0.8039 0.4353;0.4039 0.8039 0.4235;0.4196 0.8039 0.4118;0.4314 0.8039 0.4;0.4471 0.8039 0.3922;0.4627 0.8 0.3804;0.4745 0.8 0.3686;0.4902 0.8 0.3569;0.5059 0.8 0.349;0.5176 0.8 0.3373;0.5333 0.7961 0.3255;0.5451 0.7961 0.3176;0.5608 0.7961 0.3059;0.5765 0.7922 0.2941;0.5882 0.7922 0.2824;0.6039 0.7882 0.2745;0.6157 0.7882 0.2627;0.6314 0.7843 0.251;0.6431 0.7843 0.2431;0.6549 0.7804 0.2314;0.6706 0.7804 0.2235;0.6824 0.7765 0.2157;0.698 0.7765 0.2078;0.7098 0.7725 0.2;0.7216 0.7686 0.1922;0.7333 0.7686 0.1843;0.7451 0.7647 0.1765;0.7608 0.7647 0.1725;0.7725 0.7608 0.1647;0.7843 0.7569 0.1608;0.7961 0.7569 0.1569;0.8078 0.7529 0.1529;0.8157 0.749 0.1529;0.8275 0.749 0.1529;0.8392 0.7451 0.1529;0.851 0.7451 0.1569;0.8588 0.7412 0.1569;0.8706 0.7373 0.1608;0.8824 0.7373 0.1647;0.8902 0.7373 0.1686;0.902 0.7333 0.1765;0.9098 0.7333 0.1804;0.9176 0.7294 0.1882;0.9255 0.7294 0.1961;0.9373 0.7294 0.2078;0.9451 0.7294 0.2157;0.9529 0.7294 0.2235;0.9608 0.7294 0.2314;0.9686 0.7294 0.2392;0.9765 0.7294 0.2431;0.9843 0.7333 0.2431;0.9882 0.7373 0.2431;0.9961 0.7412 0.2392;0.9961 0.7451 0.2353;0.9961 0.7529 0.2314;0.9961 0.7569 0.2275;0.9961 0.7608 0.2235;0.9961 0.7686 0.2196;0.9961 0.7725 0.2157;0.9961 0.7804 0.2078;0.9961 0.7843 0.2039;0.9961 0.7922 0.2;0.9922 0.7961 0.1961;0.9922 0.8039 0.1922;0.9922 0.8078 0.1922;0.9882 0.8157 0.1882;0.9843 0.8235 0.1843;0.9843 0.8275 0.1804;0.9804 0.8353 0.1804;0.9765 0.8392 0.1765;0.9765 0.8471 0.1725;0.9725 0.851 0.1686;0.9686 0.8588 0.1647;0.9686 0.8667 0.1647;0.9647 0.8706 0.1608;0.9647 0.8784 0.1569;0.9608 0.8824 0.1569;0.9608 0.8902 0.1529;0.9608 0.898 0.149;0.9608 0.902 0.149;0.9608 0.9098 0.1451;0.9608 0.9137 0.1412;0.9608 0.9216 0.1373;0.9608 0.9255 0.1333;0.9608 0.9333 0.1294;0.9647 0.9373 0.1255;0.9647 0.9451 0.1216;0.9647 0.949 0.1176;0.9686 0.9569 0.1098;0.9686 0.9608 0.1059;0.9725 0.9686 0.102;0.9725 0.9725 0.0941;0.9765 0.9765 0.0863;0.9765 0.9843 0.0824];
                app.g1.FontSize = 11.3333333333333;
                app.g1.NextPlot = 'replace';
                app.g1.Tag = 'g1';
                app.g1.Position = [22 490 283 250];
                
                % Create g2 图像傅里叶变换
                app.g2 = uiaxes(app.uibuttongroup11);
                app.g2.FontName = 'Microsoft YaHei UI';
                app.g2.Colormap = [0.2431 0.149 0.6588;0.2431 0.1529 0.6745;0.2471 0.1569 0.6863;0.2471 0.1608 0.698;0.251 0.1647 0.7059;0.251 0.1686 0.7176;0.2549 0.1725 0.7294;0.2549 0.1765 0.7412;0.2588 0.1804 0.749;0.2588 0.1843 0.7608;0.2627 0.1922 0.7843;0.2627 0.1961 0.7922;0.2667 0.2 0.8039;0.2667 0.2039 0.8157;0.2706 0.2078 0.8235;0.2706 0.2157 0.8353;0.2706 0.2196 0.8431;0.2745 0.2235 0.851;0.2745 0.2275 0.8627;0.2745 0.2314 0.8706;0.2745 0.2392 0.8784;0.2784 0.2431 0.8824;0.2784 0.2471 0.8902;0.2784 0.2549 0.898;0.2784 0.2588 0.902;0.2784 0.2667 0.9098;0.2784 0.2706 0.9137;0.2784 0.2745 0.9216;0.2824 0.2824 0.9255;0.2824 0.2863 0.9294;0.2824 0.2941 0.9333;0.2824 0.298 0.9412;0.2824 0.3059 0.9451;0.2824 0.3098 0.949;0.2824 0.3137 0.9529;0.2824 0.3216 0.9569;0.2824 0.3255 0.9608;0.2824 0.3294 0.9647;0.2784 0.3373 0.9686;0.2784 0.3412 0.9686;0.2784 0.349 0.9725;0.2784 0.3529 0.9765;0.2784 0.3569 0.9804;0.2784 0.3647 0.9804;0.2745 0.3686 0.9843;0.2745 0.3765 0.9843;0.2745 0.3804 0.9882;0.2706 0.3843 0.9882;0.2706 0.3922 0.9922;0.2667 0.3961 0.9922;0.2627 0.4039 0.9922;0.2627 0.4078 0.9961;0.2588 0.4157 0.9961;0.2549 0.4196 0.9961;0.251 0.4275 0.9961;0.2471 0.4314 1;0.2431 0.4392 1;0.2353 0.4431 1;0.2314 0.451 1;0.2235 0.4549 1;0.2196 0.4627 0.9961;0.2118 0.4667 0.9961;0.2078 0.4745 0.9922;0.2 0.4784 0.9922;0.1961 0.4863 0.9882;0.1922 0.4902 0.9882;0.1882 0.498 0.9843;0.1843 0.502 0.9804;0.1843 0.5098 0.9804;0.1804 0.5137 0.9765;0.1804 0.5176 0.9725;0.1804 0.5255 0.9725;0.1804 0.5294 0.9686;0.1765 0.5333 0.9647;0.1765 0.5412 0.9608;0.1765 0.5451 0.9569;0.1765 0.549 0.9529;0.1765 0.5569 0.949;0.1725 0.5608 0.9451;0.1725 0.5647 0.9412;0.1686 0.5686 0.9373;0.1647 0.5765 0.9333;0.1608 0.5804 0.9294;0.1569 0.5843 0.9255;0.1529 0.5922 0.9216;0.1529 0.5961 0.9176;0.149 0.6 0.9137;0.149 0.6039 0.9098;0.1451 0.6078 0.9098;0.1451 0.6118 0.9059;0.1412 0.6196 0.902;0.1412 0.6235 0.898;0.1373 0.6275 0.898;0.1373 0.6314 0.8941;0.1333 0.6353 0.8941;0.1294 0.6392 0.8902;0.1255 0.6471 0.8902;0.1216 0.651 0.8863;0.1176 0.6549 0.8824;0.1137 0.6588 0.8824;0.1137 0.6627 0.8784;0.1098 0.6667 0.8745;0.1059 0.6706 0.8706;0.102 0.6745 0.8667;0.098 0.6784 0.8627;0.0902 0.6824 0.8549;0.0863 0.6863 0.851;0.0784 0.6902 0.8471;0.0706 0.6941 0.8392;0.0627 0.698 0.8353;0.0549 0.702 0.8314;0.0431 0.702 0.8235;0.0314 0.7059 0.8196;0.0235 0.7098 0.8118;0.0157 0.7137 0.8078;0.0078 0.7176 0.8;0.0039 0.7176 0.7922;0 0.7216 0.7882;0 0.7255 0.7804;0 0.7294 0.7765;0.0039 0.7294 0.7686;0.0078 0.7333 0.7608;0.0157 0.7333 0.7569;0.0235 0.7373 0.749;0.0353 0.7412 0.7412;0.051 0.7412 0.7373;0.0627 0.7451 0.7294;0.0784 0.7451 0.7216;0.0902 0.749 0.7137;0.102 0.7529 0.7098;0.1137 0.7529 0.702;0.1255 0.7569 0.6941;0.1373 0.7569 0.6863;0.1451 0.7608 0.6824;0.1529 0.7608 0.6745;0.1608 0.7647 0.6667;0.1686 0.7647 0.6588;0.1725 0.7686 0.651;0.1804 0.7686 0.6471;0.1843 0.7725 0.6392;0.1922 0.7725 0.6314;0.1961 0.7765 0.6235;0.2 0.7804 0.6157;0.2078 0.7804 0.6078;0.2118 0.7843 0.6;0.2196 0.7843 0.5882;0.2235 0.7882 0.5804;0.2314 0.7882 0.5725;0.2392 0.7922 0.5647;0.251 0.7922 0.5529;0.2588 0.7922 0.5451;0.2706 0.7961 0.5373;0.2824 0.7961 0.5255;0.2941 0.7961 0.5176;0.3059 0.8 0.5059;0.3176 0.8 0.498;0.3294 0.8 0.4863;0.3412 0.8 0.4784;0.3529 0.8 0.4667;0.3686 0.8039 0.4549;0.3804 0.8039 0.4471;0.3922 0.8039 0.4353;0.4039 0.8039 0.4235;0.4196 0.8039 0.4118;0.4314 0.8039 0.4;0.4471 0.8039 0.3922;0.4627 0.8 0.3804;0.4745 0.8 0.3686;0.4902 0.8 0.3569;0.5059 0.8 0.349;0.5176 0.8 0.3373;0.5333 0.7961 0.3255;0.5451 0.7961 0.3176;0.5608 0.7961 0.3059;0.5765 0.7922 0.2941;0.5882 0.7922 0.2824;0.6039 0.7882 0.2745;0.6157 0.7882 0.2627;0.6314 0.7843 0.251;0.6431 0.7843 0.2431;0.6549 0.7804 0.2314;0.6706 0.7804 0.2235;0.6824 0.7765 0.2157;0.698 0.7765 0.2078;0.7098 0.7725 0.2;0.7216 0.7686 0.1922;0.7333 0.7686 0.1843;0.7451 0.7647 0.1765;0.7608 0.7647 0.1725;0.7725 0.7608 0.1647;0.7843 0.7569 0.1608;0.7961 0.7569 0.1569;0.8078 0.7529 0.1529;0.8157 0.749 0.1529;0.8275 0.749 0.1529;0.8392 0.7451 0.1529;0.851 0.7451 0.1569;0.8588 0.7412 0.1569;0.8706 0.7373 0.1608;0.8824 0.7373 0.1647;0.8902 0.7373 0.1686;0.902 0.7333 0.1765;0.9098 0.7333 0.1804;0.9176 0.7294 0.1882;0.9255 0.7294 0.1961;0.9373 0.7294 0.2078;0.9451 0.7294 0.2157;0.9529 0.7294 0.2235;0.9608 0.7294 0.2314;0.9686 0.7294 0.2392;0.9765 0.7294 0.2431;0.9843 0.7333 0.2431;0.9882 0.7373 0.2431;0.9961 0.7412 0.2392;0.9961 0.7451 0.2353;0.9961 0.7529 0.2314;0.9961 0.7569 0.2275;0.9961 0.7608 0.2235;0.9961 0.7686 0.2196;0.9961 0.7725 0.2157;0.9961 0.7804 0.2078;0.9961 0.7843 0.2039;0.9961 0.7922 0.2;0.9922 0.7961 0.1961;0.9922 0.8039 0.1922;0.9922 0.8078 0.1922;0.9882 0.8157 0.1882;0.9843 0.8235 0.1843;0.9843 0.8275 0.1804;0.9804 0.8353 0.1804;0.9765 0.8392 0.1765;0.9765 0.8471 0.1725;0.9725 0.851 0.1686;0.9686 0.8588 0.1647;0.9686 0.8667 0.1647;0.9647 0.8706 0.1608;0.9647 0.8784 0.1569;0.9608 0.8824 0.1569;0.9608 0.8902 0.1529;0.9608 0.898 0.149;0.9608 0.902 0.149;0.9608 0.9098 0.1451;0.9608 0.9137 0.1412;0.9608 0.9216 0.1373;0.9608 0.9255 0.1333;0.9608 0.9333 0.1294;0.9647 0.9373 0.1255;0.9647 0.9451 0.1216;0.9647 0.949 0.1176;0.9686 0.9569 0.1098;0.9686 0.9608 0.1059;0.9725 0.9686 0.102;0.9725 0.9725 0.0941;0.9765 0.9765 0.0863;0.9765 0.9843 0.0824];
                app.g2.FontSize = 11.3333333333333;
                app.g2.NextPlot = 'replace';
                app.g2.Tag = 'g2';
                app.g2.Position = [70 240 283 250];
                
                % Create g3 DCT变换
                app.g3 = uiaxes(app.uibuttongroup11);
                app.g3.FontName = 'Microsoft YaHei UI';
                app.g3.Colormap = [0.2431 0.149 0.6588;0.2431 0.1529 0.6745;0.2471 0.1569 0.6863;0.2471 0.1608 0.698;0.251 0.1647 0.7059;0.251 0.1686 0.7176;0.2549 0.1725 0.7294;0.2549 0.1765 0.7412;0.2588 0.1804 0.749;0.2588 0.1843 0.7608;0.2627 0.1922 0.7843;0.2627 0.1961 0.7922;0.2667 0.2 0.8039;0.2667 0.2039 0.8157;0.2706 0.2078 0.8235;0.2706 0.2157 0.8353;0.2706 0.2196 0.8431;0.2745 0.2235 0.851;0.2745 0.2275 0.8627;0.2745 0.2314 0.8706;0.2745 0.2392 0.8784;0.2784 0.2431 0.8824;0.2784 0.2471 0.8902;0.2784 0.2549 0.898;0.2784 0.2588 0.902;0.2784 0.2667 0.9098;0.2784 0.2706 0.9137;0.2784 0.2745 0.9216;0.2824 0.2824 0.9255;0.2824 0.2863 0.9294;0.2824 0.2941 0.9333;0.2824 0.298 0.9412;0.2824 0.3059 0.9451;0.2824 0.3098 0.949;0.2824 0.3137 0.9529;0.2824 0.3216 0.9569;0.2824 0.3255 0.9608;0.2824 0.3294 0.9647;0.2784 0.3373 0.9686;0.2784 0.3412 0.9686;0.2784 0.349 0.9725;0.2784 0.3529 0.9765;0.2784 0.3569 0.9804;0.2784 0.3647 0.9804;0.2745 0.3686 0.9843;0.2745 0.3765 0.9843;0.2745 0.3804 0.9882;0.2706 0.3843 0.9882;0.2706 0.3922 0.9922;0.2667 0.3961 0.9922;0.2627 0.4039 0.9922;0.2627 0.4078 0.9961;0.2588 0.4157 0.9961;0.2549 0.4196 0.9961;0.251 0.4275 0.9961;0.2471 0.4314 1;0.2431 0.4392 1;0.2353 0.4431 1;0.2314 0.451 1;0.2235 0.4549 1;0.2196 0.4627 0.9961;0.2118 0.4667 0.9961;0.2078 0.4745 0.9922;0.2 0.4784 0.9922;0.1961 0.4863 0.9882;0.1922 0.4902 0.9882;0.1882 0.498 0.9843;0.1843 0.502 0.9804;0.1843 0.5098 0.9804;0.1804 0.5137 0.9765;0.1804 0.5176 0.9725;0.1804 0.5255 0.9725;0.1804 0.5294 0.9686;0.1765 0.5333 0.9647;0.1765 0.5412 0.9608;0.1765 0.5451 0.9569;0.1765 0.549 0.9529;0.1765 0.5569 0.949;0.1725 0.5608 0.9451;0.1725 0.5647 0.9412;0.1686 0.5686 0.9373;0.1647 0.5765 0.9333;0.1608 0.5804 0.9294;0.1569 0.5843 0.9255;0.1529 0.5922 0.9216;0.1529 0.5961 0.9176;0.149 0.6 0.9137;0.149 0.6039 0.9098;0.1451 0.6078 0.9098;0.1451 0.6118 0.9059;0.1412 0.6196 0.902;0.1412 0.6235 0.898;0.1373 0.6275 0.898;0.1373 0.6314 0.8941;0.1333 0.6353 0.8941;0.1294 0.6392 0.8902;0.1255 0.6471 0.8902;0.1216 0.651 0.8863;0.1176 0.6549 0.8824;0.1137 0.6588 0.8824;0.1137 0.6627 0.8784;0.1098 0.6667 0.8745;0.1059 0.6706 0.8706;0.102 0.6745 0.8667;0.098 0.6784 0.8627;0.0902 0.6824 0.8549;0.0863 0.6863 0.851;0.0784 0.6902 0.8471;0.0706 0.6941 0.8392;0.0627 0.698 0.8353;0.0549 0.702 0.8314;0.0431 0.702 0.8235;0.0314 0.7059 0.8196;0.0235 0.7098 0.8118;0.0157 0.7137 0.8078;0.0078 0.7176 0.8;0.0039 0.7176 0.7922;0 0.7216 0.7882;0 0.7255 0.7804;0 0.7294 0.7765;0.0039 0.7294 0.7686;0.0078 0.7333 0.7608;0.0157 0.7333 0.7569;0.0235 0.7373 0.749;0.0353 0.7412 0.7412;0.051 0.7412 0.7373;0.0627 0.7451 0.7294;0.0784 0.7451 0.7216;0.0902 0.749 0.7137;0.102 0.7529 0.7098;0.1137 0.7529 0.702;0.1255 0.7569 0.6941;0.1373 0.7569 0.6863;0.1451 0.7608 0.6824;0.1529 0.7608 0.6745;0.1608 0.7647 0.6667;0.1686 0.7647 0.6588;0.1725 0.7686 0.651;0.1804 0.7686 0.6471;0.1843 0.7725 0.6392;0.1922 0.7725 0.6314;0.1961 0.7765 0.6235;0.2 0.7804 0.6157;0.2078 0.7804 0.6078;0.2118 0.7843 0.6;0.2196 0.7843 0.5882;0.2235 0.7882 0.5804;0.2314 0.7882 0.5725;0.2392 0.7922 0.5647;0.251 0.7922 0.5529;0.2588 0.7922 0.5451;0.2706 0.7961 0.5373;0.2824 0.7961 0.5255;0.2941 0.7961 0.5176;0.3059 0.8 0.5059;0.3176 0.8 0.498;0.3294 0.8 0.4863;0.3412 0.8 0.4784;0.3529 0.8 0.4667;0.3686 0.8039 0.4549;0.3804 0.8039 0.4471;0.3922 0.8039 0.4353;0.4039 0.8039 0.4235;0.4196 0.8039 0.4118;0.4314 0.8039 0.4;0.4471 0.8039 0.3922;0.4627 0.8 0.3804;0.4745 0.8 0.3686;0.4902 0.8 0.3569;0.5059 0.8 0.349;0.5176 0.8 0.3373;0.5333 0.7961 0.3255;0.5451 0.7961 0.3176;0.5608 0.7961 0.3059;0.5765 0.7922 0.2941;0.5882 0.7922 0.2824;0.6039 0.7882 0.2745;0.6157 0.7882 0.2627;0.6314 0.7843 0.251;0.6431 0.7843 0.2431;0.6549 0.7804 0.2314;0.6706 0.7804 0.2235;0.6824 0.7765 0.2157;0.698 0.7765 0.2078;0.7098 0.7725 0.2;0.7216 0.7686 0.1922;0.7333 0.7686 0.1843;0.7451 0.7647 0.1765;0.7608 0.7647 0.1725;0.7725 0.7608 0.1647;0.7843 0.7569 0.1608;0.7961 0.7569 0.1569;0.8078 0.7529 0.1529;0.8157 0.749 0.1529;0.8275 0.749 0.1529;0.8392 0.7451 0.1529;0.851 0.7451 0.1569;0.8588 0.7412 0.1569;0.8706 0.7373 0.1608;0.8824 0.7373 0.1647;0.8902 0.7373 0.1686;0.902 0.7333 0.1765;0.9098 0.7333 0.1804;0.9176 0.7294 0.1882;0.9255 0.7294 0.1961;0.9373 0.7294 0.2078;0.9451 0.7294 0.2157;0.9529 0.7294 0.2235;0.9608 0.7294 0.2314;0.9686 0.7294 0.2392;0.9765 0.7294 0.2431;0.9843 0.7333 0.2431;0.9882 0.7373 0.2431;0.9961 0.7412 0.2392;0.9961 0.7451 0.2353;0.9961 0.7529 0.2314;0.9961 0.7569 0.2275;0.9961 0.7608 0.2235;0.9961 0.7686 0.2196;0.9961 0.7725 0.2157;0.9961 0.7804 0.2078;0.9961 0.7843 0.2039;0.9961 0.7922 0.2;0.9922 0.7961 0.1961;0.9922 0.8039 0.1922;0.9922 0.8078 0.1922;0.9882 0.8157 0.1882;0.9843 0.8235 0.1843;0.9843 0.8275 0.1804;0.9804 0.8353 0.1804;0.9765 0.8392 0.1765;0.9765 0.8471 0.1725;0.9725 0.851 0.1686;0.9686 0.8588 0.1647;0.9686 0.8667 0.1647;0.9647 0.8706 0.1608;0.9647 0.8784 0.1569;0.9608 0.8824 0.1569;0.9608 0.8902 0.1529;0.9608 0.898 0.149;0.9608 0.902 0.149;0.9608 0.9098 0.1451;0.9608 0.9137 0.1412;0.9608 0.9216 0.1373;0.9608 0.9255 0.1333;0.9608 0.9333 0.1294;0.9647 0.9373 0.1255;0.9647 0.9451 0.1216;0.9647 0.949 0.1176;0.9686 0.9569 0.1098;0.9686 0.9608 0.1059;0.9725 0.9686 0.102;0.9725 0.9725 0.0941;0.9765 0.9765 0.0863;0.9765 0.9843 0.0824];
                app.g3.FontSize = 11.3333333333333;
                app.g3.NextPlot = 'replace';
                app.g3.Tag = 'g3';
                app.g3.Position = [70 0 283 250];
                
                % Create g4 小波变换1
                app.g4 = uiaxes(app.uibuttongroup11);
                app.g4.FontName = 'Microsoft YaHei UI';
                app.g4.Colormap = [0.2431 0.149 0.6588;0.2431 0.1529 0.6745;0.2471 0.1569 0.6863;0.2471 0.1608 0.698;0.251 0.1647 0.7059;0.251 0.1686 0.7176;0.2549 0.1725 0.7294;0.2549 0.1765 0.7412;0.2588 0.1804 0.749;0.2588 0.1843 0.7608;0.2627 0.1922 0.7843;0.2627 0.1961 0.7922;0.2667 0.2 0.8039;0.2667 0.2039 0.8157;0.2706 0.2078 0.8235;0.2706 0.2157 0.8353;0.2706 0.2196 0.8431;0.2745 0.2235 0.851;0.2745 0.2275 0.8627;0.2745 0.2314 0.8706;0.2745 0.2392 0.8784;0.2784 0.2431 0.8824;0.2784 0.2471 0.8902;0.2784 0.2549 0.898;0.2784 0.2588 0.902;0.2784 0.2667 0.9098;0.2784 0.2706 0.9137;0.2784 0.2745 0.9216;0.2824 0.2824 0.9255;0.2824 0.2863 0.9294;0.2824 0.2941 0.9333;0.2824 0.298 0.9412;0.2824 0.3059 0.9451;0.2824 0.3098 0.949;0.2824 0.3137 0.9529;0.2824 0.3216 0.9569;0.2824 0.3255 0.9608;0.2824 0.3294 0.9647;0.2784 0.3373 0.9686;0.2784 0.3412 0.9686;0.2784 0.349 0.9725;0.2784 0.3529 0.9765;0.2784 0.3569 0.9804;0.2784 0.3647 0.9804;0.2745 0.3686 0.9843;0.2745 0.3765 0.9843;0.2745 0.3804 0.9882;0.2706 0.3843 0.9882;0.2706 0.3922 0.9922;0.2667 0.3961 0.9922;0.2627 0.4039 0.9922;0.2627 0.4078 0.9961;0.2588 0.4157 0.9961;0.2549 0.4196 0.9961;0.251 0.4275 0.9961;0.2471 0.4314 1;0.2431 0.4392 1;0.2353 0.4431 1;0.2314 0.451 1;0.2235 0.4549 1;0.2196 0.4627 0.9961;0.2118 0.4667 0.9961;0.2078 0.4745 0.9922;0.2 0.4784 0.9922;0.1961 0.4863 0.9882;0.1922 0.4902 0.9882;0.1882 0.498 0.9843;0.1843 0.502 0.9804;0.1843 0.5098 0.9804;0.1804 0.5137 0.9765;0.1804 0.5176 0.9725;0.1804 0.5255 0.9725;0.1804 0.5294 0.9686;0.1765 0.5333 0.9647;0.1765 0.5412 0.9608;0.1765 0.5451 0.9569;0.1765 0.549 0.9529;0.1765 0.5569 0.949;0.1725 0.5608 0.9451;0.1725 0.5647 0.9412;0.1686 0.5686 0.9373;0.1647 0.5765 0.9333;0.1608 0.5804 0.9294;0.1569 0.5843 0.9255;0.1529 0.5922 0.9216;0.1529 0.5961 0.9176;0.149 0.6 0.9137;0.149 0.6039 0.9098;0.1451 0.6078 0.9098;0.1451 0.6118 0.9059;0.1412 0.6196 0.902;0.1412 0.6235 0.898;0.1373 0.6275 0.898;0.1373 0.6314 0.8941;0.1333 0.6353 0.8941;0.1294 0.6392 0.8902;0.1255 0.6471 0.8902;0.1216 0.651 0.8863;0.1176 0.6549 0.8824;0.1137 0.6588 0.8824;0.1137 0.6627 0.8784;0.1098 0.6667 0.8745;0.1059 0.6706 0.8706;0.102 0.6745 0.8667;0.098 0.6784 0.8627;0.0902 0.6824 0.8549;0.0863 0.6863 0.851;0.0784 0.6902 0.8471;0.0706 0.6941 0.8392;0.0627 0.698 0.8353;0.0549 0.702 0.8314;0.0431 0.702 0.8235;0.0314 0.7059 0.8196;0.0235 0.7098 0.8118;0.0157 0.7137 0.8078;0.0078 0.7176 0.8;0.0039 0.7176 0.7922;0 0.7216 0.7882;0 0.7255 0.7804;0 0.7294 0.7765;0.0039 0.7294 0.7686;0.0078 0.7333 0.7608;0.0157 0.7333 0.7569;0.0235 0.7373 0.749;0.0353 0.7412 0.7412;0.051 0.7412 0.7373;0.0627 0.7451 0.7294;0.0784 0.7451 0.7216;0.0902 0.749 0.7137;0.102 0.7529 0.7098;0.1137 0.7529 0.702;0.1255 0.7569 0.6941;0.1373 0.7569 0.6863;0.1451 0.7608 0.6824;0.1529 0.7608 0.6745;0.1608 0.7647 0.6667;0.1686 0.7647 0.6588;0.1725 0.7686 0.651;0.1804 0.7686 0.6471;0.1843 0.7725 0.6392;0.1922 0.7725 0.6314;0.1961 0.7765 0.6235;0.2 0.7804 0.6157;0.2078 0.7804 0.6078;0.2118 0.7843 0.6;0.2196 0.7843 0.5882;0.2235 0.7882 0.5804;0.2314 0.7882 0.5725;0.2392 0.7922 0.5647;0.251 0.7922 0.5529;0.2588 0.7922 0.5451;0.2706 0.7961 0.5373;0.2824 0.7961 0.5255;0.2941 0.7961 0.5176;0.3059 0.8 0.5059;0.3176 0.8 0.498;0.3294 0.8 0.4863;0.3412 0.8 0.4784;0.3529 0.8 0.4667;0.3686 0.8039 0.4549;0.3804 0.8039 0.4471;0.3922 0.8039 0.4353;0.4039 0.8039 0.4235;0.4196 0.8039 0.4118;0.4314 0.8039 0.4;0.4471 0.8039 0.3922;0.4627 0.8 0.3804;0.4745 0.8 0.3686;0.4902 0.8 0.3569;0.5059 0.8 0.349;0.5176 0.8 0.3373;0.5333 0.7961 0.3255;0.5451 0.7961 0.3176;0.5608 0.7961 0.3059;0.5765 0.7922 0.2941;0.5882 0.7922 0.2824;0.6039 0.7882 0.2745;0.6157 0.7882 0.2627;0.6314 0.7843 0.251;0.6431 0.7843 0.2431;0.6549 0.7804 0.2314;0.6706 0.7804 0.2235;0.6824 0.7765 0.2157;0.698 0.7765 0.2078;0.7098 0.7725 0.2;0.7216 0.7686 0.1922;0.7333 0.7686 0.1843;0.7451 0.7647 0.1765;0.7608 0.7647 0.1725;0.7725 0.7608 0.1647;0.7843 0.7569 0.1608;0.7961 0.7569 0.1569;0.8078 0.7529 0.1529;0.8157 0.749 0.1529;0.8275 0.749 0.1529;0.8392 0.7451 0.1529;0.851 0.7451 0.1569;0.8588 0.7412 0.1569;0.8706 0.7373 0.1608;0.8824 0.7373 0.1647;0.8902 0.7373 0.1686;0.902 0.7333 0.1765;0.9098 0.7333 0.1804;0.9176 0.7294 0.1882;0.9255 0.7294 0.1961;0.9373 0.7294 0.2078;0.9451 0.7294 0.2157;0.9529 0.7294 0.2235;0.9608 0.7294 0.2314;0.9686 0.7294 0.2392;0.9765 0.7294 0.2431;0.9843 0.7333 0.2431;0.9882 0.7373 0.2431;0.9961 0.7412 0.2392;0.9961 0.7451 0.2353;0.9961 0.7529 0.2314;0.9961 0.7569 0.2275;0.9961 0.7608 0.2235;0.9961 0.7686 0.2196;0.9961 0.7725 0.2157;0.9961 0.7804 0.2078;0.9961 0.7843 0.2039;0.9961 0.7922 0.2;0.9922 0.7961 0.1961;0.9922 0.8039 0.1922;0.9922 0.8078 0.1922;0.9882 0.8157 0.1882;0.9843 0.8235 0.1843;0.9843 0.8275 0.1804;0.9804 0.8353 0.1804;0.9765 0.8392 0.1765;0.9765 0.8471 0.1725;0.9725 0.851 0.1686;0.9686 0.8588 0.1647;0.9686 0.8667 0.1647;0.9647 0.8706 0.1608;0.9647 0.8784 0.1569;0.9608 0.8824 0.1569;0.9608 0.8902 0.1529;0.9608 0.898 0.149;0.9608 0.902 0.149;0.9608 0.9098 0.1451;0.9608 0.9137 0.1412;0.9608 0.9216 0.1373;0.9608 0.9255 0.1333;0.9608 0.9333 0.1294;0.9647 0.9373 0.1255;0.9647 0.9451 0.1216;0.9647 0.949 0.1176;0.9686 0.9569 0.1098;0.9686 0.9608 0.1059;0.9725 0.9686 0.102;0.9725 0.9725 0.0941;0.9765 0.9765 0.0863;0.9765 0.9843 0.0824];
                app.g4.FontSize = 11.3333333333333;
                app.g4.NextPlot = 'replace';
                app.g4.Tag = 'g4';
                app.g4.Position = [440 550 273 200];
                
                % Create g5 小波变换2
                app.g5 = uiaxes(app.uibuttongroup11);
                app.g5.FontName = 'Microsoft YaHei UI';
                app.g5.Colormap = [0.2431 0.149 0.6588;0.2431 0.1529 0.6745;0.2471 0.1569 0.6863;0.2471 0.1608 0.698;0.251 0.1647 0.7059;0.251 0.1686 0.7176;0.2549 0.1725 0.7294;0.2549 0.1765 0.7412;0.2588 0.1804 0.749;0.2588 0.1843 0.7608;0.2627 0.1922 0.7843;0.2627 0.1961 0.7922;0.2667 0.2 0.8039;0.2667 0.2039 0.8157;0.2706 0.2078 0.8235;0.2706 0.2157 0.8353;0.2706 0.2196 0.8431;0.2745 0.2235 0.851;0.2745 0.2275 0.8627;0.2745 0.2314 0.8706;0.2745 0.2392 0.8784;0.2784 0.2431 0.8824;0.2784 0.2471 0.8902;0.2784 0.2549 0.898;0.2784 0.2588 0.902;0.2784 0.2667 0.9098;0.2784 0.2706 0.9137;0.2784 0.2745 0.9216;0.2824 0.2824 0.9255;0.2824 0.2863 0.9294;0.2824 0.2941 0.9333;0.2824 0.298 0.9412;0.2824 0.3059 0.9451;0.2824 0.3098 0.949;0.2824 0.3137 0.9529;0.2824 0.3216 0.9569;0.2824 0.3255 0.9608;0.2824 0.3294 0.9647;0.2784 0.3373 0.9686;0.2784 0.3412 0.9686;0.2784 0.349 0.9725;0.2784 0.3529 0.9765;0.2784 0.3569 0.9804;0.2784 0.3647 0.9804;0.2745 0.3686 0.9843;0.2745 0.3765 0.9843;0.2745 0.3804 0.9882;0.2706 0.3843 0.9882;0.2706 0.3922 0.9922;0.2667 0.3961 0.9922;0.2627 0.4039 0.9922;0.2627 0.4078 0.9961;0.2588 0.4157 0.9961;0.2549 0.4196 0.9961;0.251 0.4275 0.9961;0.2471 0.4314 1;0.2431 0.4392 1;0.2353 0.4431 1;0.2314 0.451 1;0.2235 0.4549 1;0.2196 0.4627 0.9961;0.2118 0.4667 0.9961;0.2078 0.4745 0.9922;0.2 0.4784 0.9922;0.1961 0.4863 0.9882;0.1922 0.4902 0.9882;0.1882 0.498 0.9843;0.1843 0.502 0.9804;0.1843 0.5098 0.9804;0.1804 0.5137 0.9765;0.1804 0.5176 0.9725;0.1804 0.5255 0.9725;0.1804 0.5294 0.9686;0.1765 0.5333 0.9647;0.1765 0.5412 0.9608;0.1765 0.5451 0.9569;0.1765 0.549 0.9529;0.1765 0.5569 0.949;0.1725 0.5608 0.9451;0.1725 0.5647 0.9412;0.1686 0.5686 0.9373;0.1647 0.5765 0.9333;0.1608 0.5804 0.9294;0.1569 0.5843 0.9255;0.1529 0.5922 0.9216;0.1529 0.5961 0.9176;0.149 0.6 0.9137;0.149 0.6039 0.9098;0.1451 0.6078 0.9098;0.1451 0.6118 0.9059;0.1412 0.6196 0.902;0.1412 0.6235 0.898;0.1373 0.6275 0.898;0.1373 0.6314 0.8941;0.1333 0.6353 0.8941;0.1294 0.6392 0.8902;0.1255 0.6471 0.8902;0.1216 0.651 0.8863;0.1176 0.6549 0.8824;0.1137 0.6588 0.8824;0.1137 0.6627 0.8784;0.1098 0.6667 0.8745;0.1059 0.6706 0.8706;0.102 0.6745 0.8667;0.098 0.6784 0.8627;0.0902 0.6824 0.8549;0.0863 0.6863 0.851;0.0784 0.6902 0.8471;0.0706 0.6941 0.8392;0.0627 0.698 0.8353;0.0549 0.702 0.8314;0.0431 0.702 0.8235;0.0314 0.7059 0.8196;0.0235 0.7098 0.8118;0.0157 0.7137 0.8078;0.0078 0.7176 0.8;0.0039 0.7176 0.7922;0 0.7216 0.7882;0 0.7255 0.7804;0 0.7294 0.7765;0.0039 0.7294 0.7686;0.0078 0.7333 0.7608;0.0157 0.7333 0.7569;0.0235 0.7373 0.749;0.0353 0.7412 0.7412;0.051 0.7412 0.7373;0.0627 0.7451 0.7294;0.0784 0.7451 0.7216;0.0902 0.749 0.7137;0.102 0.7529 0.7098;0.1137 0.7529 0.702;0.1255 0.7569 0.6941;0.1373 0.7569 0.6863;0.1451 0.7608 0.6824;0.1529 0.7608 0.6745;0.1608 0.7647 0.6667;0.1686 0.7647 0.6588;0.1725 0.7686 0.651;0.1804 0.7686 0.6471;0.1843 0.7725 0.6392;0.1922 0.7725 0.6314;0.1961 0.7765 0.6235;0.2 0.7804 0.6157;0.2078 0.7804 0.6078;0.2118 0.7843 0.6;0.2196 0.7843 0.5882;0.2235 0.7882 0.5804;0.2314 0.7882 0.5725;0.2392 0.7922 0.5647;0.251 0.7922 0.5529;0.2588 0.7922 0.5451;0.2706 0.7961 0.5373;0.2824 0.7961 0.5255;0.2941 0.7961 0.5176;0.3059 0.8 0.5059;0.3176 0.8 0.498;0.3294 0.8 0.4863;0.3412 0.8 0.4784;0.3529 0.8 0.4667;0.3686 0.8039 0.4549;0.3804 0.8039 0.4471;0.3922 0.8039 0.4353;0.4039 0.8039 0.4235;0.4196 0.8039 0.4118;0.4314 0.8039 0.4;0.4471 0.8039 0.3922;0.4627 0.8 0.3804;0.4745 0.8 0.3686;0.4902 0.8 0.3569;0.5059 0.8 0.349;0.5176 0.8 0.3373;0.5333 0.7961 0.3255;0.5451 0.7961 0.3176;0.5608 0.7961 0.3059;0.5765 0.7922 0.2941;0.5882 0.7922 0.2824;0.6039 0.7882 0.2745;0.6157 0.7882 0.2627;0.6314 0.7843 0.251;0.6431 0.7843 0.2431;0.6549 0.7804 0.2314;0.6706 0.7804 0.2235;0.6824 0.7765 0.2157;0.698 0.7765 0.2078;0.7098 0.7725 0.2;0.7216 0.7686 0.1922;0.7333 0.7686 0.1843;0.7451 0.7647 0.1765;0.7608 0.7647 0.1725;0.7725 0.7608 0.1647;0.7843 0.7569 0.1608;0.7961 0.7569 0.1569;0.8078 0.7529 0.1529;0.8157 0.749 0.1529;0.8275 0.749 0.1529;0.8392 0.7451 0.1529;0.851 0.7451 0.1569;0.8588 0.7412 0.1569;0.8706 0.7373 0.1608;0.8824 0.7373 0.1647;0.8902 0.7373 0.1686;0.902 0.7333 0.1765;0.9098 0.7333 0.1804;0.9176 0.7294 0.1882;0.9255 0.7294 0.1961;0.9373 0.7294 0.2078;0.9451 0.7294 0.2157;0.9529 0.7294 0.2235;0.9608 0.7294 0.2314;0.9686 0.7294 0.2392;0.9765 0.7294 0.2431;0.9843 0.7333 0.2431;0.9882 0.7373 0.2431;0.9961 0.7412 0.2392;0.9961 0.7451 0.2353;0.9961 0.7529 0.2314;0.9961 0.7569 0.2275;0.9961 0.7608 0.2235;0.9961 0.7686 0.2196;0.9961 0.7725 0.2157;0.9961 0.7804 0.2078;0.9961 0.7843 0.2039;0.9961 0.7922 0.2;0.9922 0.7961 0.1961;0.9922 0.8039 0.1922;0.9922 0.8078 0.1922;0.9882 0.8157 0.1882;0.9843 0.8235 0.1843;0.9843 0.8275 0.1804;0.9804 0.8353 0.1804;0.9765 0.8392 0.1765;0.9765 0.8471 0.1725;0.9725 0.851 0.1686;0.9686 0.8588 0.1647;0.9686 0.8667 0.1647;0.9647 0.8706 0.1608;0.9647 0.8784 0.1569;0.9608 0.8824 0.1569;0.9608 0.8902 0.1529;0.9608 0.898 0.149;0.9608 0.902 0.149;0.9608 0.9098 0.1451;0.9608 0.9137 0.1412;0.9608 0.9216 0.1373;0.9608 0.9255 0.1333;0.9608 0.9333 0.1294;0.9647 0.9373 0.1255;0.9647 0.9451 0.1216;0.9647 0.949 0.1176;0.9686 0.9569 0.1098;0.9686 0.9608 0.1059;0.9725 0.9686 0.102;0.9725 0.9725 0.0941;0.9765 0.9765 0.0863;0.9765 0.9843 0.0824];
                app.g5.FontSize = 11.3333333333333;
                app.g5.NextPlot = 'replace';
                app.g5.Tag = 'g5';
                app.g5.Position = [440 370 273 200];
                
                % Create g6 小波变换3
                app.g6 = uiaxes(app.uibuttongroup11);
                app.g6.FontName = 'Microsoft YaHei UI';
                app.g6.Colormap = [0.2431 0.149 0.6588;0.2431 0.1529 0.6745;0.2471 0.1569 0.6863;0.2471 0.1608 0.698;0.251 0.1647 0.7059;0.251 0.1686 0.7176;0.2549 0.1725 0.7294;0.2549 0.1765 0.7412;0.2588 0.1804 0.749;0.2588 0.1843 0.7608;0.2627 0.1922 0.7843;0.2627 0.1961 0.7922;0.2667 0.2 0.8039;0.2667 0.2039 0.8157;0.2706 0.2078 0.8235;0.2706 0.2157 0.8353;0.2706 0.2196 0.8431;0.2745 0.2235 0.851;0.2745 0.2275 0.8627;0.2745 0.2314 0.8706;0.2745 0.2392 0.8784;0.2784 0.2431 0.8824;0.2784 0.2471 0.8902;0.2784 0.2549 0.898;0.2784 0.2588 0.902;0.2784 0.2667 0.9098;0.2784 0.2706 0.9137;0.2784 0.2745 0.9216;0.2824 0.2824 0.9255;0.2824 0.2863 0.9294;0.2824 0.2941 0.9333;0.2824 0.298 0.9412;0.2824 0.3059 0.9451;0.2824 0.3098 0.949;0.2824 0.3137 0.9529;0.2824 0.3216 0.9569;0.2824 0.3255 0.9608;0.2824 0.3294 0.9647;0.2784 0.3373 0.9686;0.2784 0.3412 0.9686;0.2784 0.349 0.9725;0.2784 0.3529 0.9765;0.2784 0.3569 0.9804;0.2784 0.3647 0.9804;0.2745 0.3686 0.9843;0.2745 0.3765 0.9843;0.2745 0.3804 0.9882;0.2706 0.3843 0.9882;0.2706 0.3922 0.9922;0.2667 0.3961 0.9922;0.2627 0.4039 0.9922;0.2627 0.4078 0.9961;0.2588 0.4157 0.9961;0.2549 0.4196 0.9961;0.251 0.4275 0.9961;0.2471 0.4314 1;0.2431 0.4392 1;0.2353 0.4431 1;0.2314 0.451 1;0.2235 0.4549 1;0.2196 0.4627 0.9961;0.2118 0.4667 0.9961;0.2078 0.4745 0.9922;0.2 0.4784 0.9922;0.1961 0.4863 0.9882;0.1922 0.4902 0.9882;0.1882 0.498 0.9843;0.1843 0.502 0.9804;0.1843 0.5098 0.9804;0.1804 0.5137 0.9765;0.1804 0.5176 0.9725;0.1804 0.5255 0.9725;0.1804 0.5294 0.9686;0.1765 0.5333 0.9647;0.1765 0.5412 0.9608;0.1765 0.5451 0.9569;0.1765 0.549 0.9529;0.1765 0.5569 0.949;0.1725 0.5608 0.9451;0.1725 0.5647 0.9412;0.1686 0.5686 0.9373;0.1647 0.5765 0.9333;0.1608 0.5804 0.9294;0.1569 0.5843 0.9255;0.1529 0.5922 0.9216;0.1529 0.5961 0.9176;0.149 0.6 0.9137;0.149 0.6039 0.9098;0.1451 0.6078 0.9098;0.1451 0.6118 0.9059;0.1412 0.6196 0.902;0.1412 0.6235 0.898;0.1373 0.6275 0.898;0.1373 0.6314 0.8941;0.1333 0.6353 0.8941;0.1294 0.6392 0.8902;0.1255 0.6471 0.8902;0.1216 0.651 0.8863;0.1176 0.6549 0.8824;0.1137 0.6588 0.8824;0.1137 0.6627 0.8784;0.1098 0.6667 0.8745;0.1059 0.6706 0.8706;0.102 0.6745 0.8667;0.098 0.6784 0.8627;0.0902 0.6824 0.8549;0.0863 0.6863 0.851;0.0784 0.6902 0.8471;0.0706 0.6941 0.8392;0.0627 0.698 0.8353;0.0549 0.702 0.8314;0.0431 0.702 0.8235;0.0314 0.7059 0.8196;0.0235 0.7098 0.8118;0.0157 0.7137 0.8078;0.0078 0.7176 0.8;0.0039 0.7176 0.7922;0 0.7216 0.7882;0 0.7255 0.7804;0 0.7294 0.7765;0.0039 0.7294 0.7686;0.0078 0.7333 0.7608;0.0157 0.7333 0.7569;0.0235 0.7373 0.749;0.0353 0.7412 0.7412;0.051 0.7412 0.7373;0.0627 0.7451 0.7294;0.0784 0.7451 0.7216;0.0902 0.749 0.7137;0.102 0.7529 0.7098;0.1137 0.7529 0.702;0.1255 0.7569 0.6941;0.1373 0.7569 0.6863;0.1451 0.7608 0.6824;0.1529 0.7608 0.6745;0.1608 0.7647 0.6667;0.1686 0.7647 0.6588;0.1725 0.7686 0.651;0.1804 0.7686 0.6471;0.1843 0.7725 0.6392;0.1922 0.7725 0.6314;0.1961 0.7765 0.6235;0.2 0.7804 0.6157;0.2078 0.7804 0.6078;0.2118 0.7843 0.6;0.2196 0.7843 0.5882;0.2235 0.7882 0.5804;0.2314 0.7882 0.5725;0.2392 0.7922 0.5647;0.251 0.7922 0.5529;0.2588 0.7922 0.5451;0.2706 0.7961 0.5373;0.2824 0.7961 0.5255;0.2941 0.7961 0.5176;0.3059 0.8 0.5059;0.3176 0.8 0.498;0.3294 0.8 0.4863;0.3412 0.8 0.4784;0.3529 0.8 0.4667;0.3686 0.8039 0.4549;0.3804 0.8039 0.4471;0.3922 0.8039 0.4353;0.4039 0.8039 0.4235;0.4196 0.8039 0.4118;0.4314 0.8039 0.4;0.4471 0.8039 0.3922;0.4627 0.8 0.3804;0.4745 0.8 0.3686;0.4902 0.8 0.3569;0.5059 0.8 0.349;0.5176 0.8 0.3373;0.5333 0.7961 0.3255;0.5451 0.7961 0.3176;0.5608 0.7961 0.3059;0.5765 0.7922 0.2941;0.5882 0.7922 0.2824;0.6039 0.7882 0.2745;0.6157 0.7882 0.2627;0.6314 0.7843 0.251;0.6431 0.7843 0.2431;0.6549 0.7804 0.2314;0.6706 0.7804 0.2235;0.6824 0.7765 0.2157;0.698 0.7765 0.2078;0.7098 0.7725 0.2;0.7216 0.7686 0.1922;0.7333 0.7686 0.1843;0.7451 0.7647 0.1765;0.7608 0.7647 0.1725;0.7725 0.7608 0.1647;0.7843 0.7569 0.1608;0.7961 0.7569 0.1569;0.8078 0.7529 0.1529;0.8157 0.749 0.1529;0.8275 0.749 0.1529;0.8392 0.7451 0.1529;0.851 0.7451 0.1569;0.8588 0.7412 0.1569;0.8706 0.7373 0.1608;0.8824 0.7373 0.1647;0.8902 0.7373 0.1686;0.902 0.7333 0.1765;0.9098 0.7333 0.1804;0.9176 0.7294 0.1882;0.9255 0.7294 0.1961;0.9373 0.7294 0.2078;0.9451 0.7294 0.2157;0.9529 0.7294 0.2235;0.9608 0.7294 0.2314;0.9686 0.7294 0.2392;0.9765 0.7294 0.2431;0.9843 0.7333 0.2431;0.9882 0.7373 0.2431;0.9961 0.7412 0.2392;0.9961 0.7451 0.2353;0.9961 0.7529 0.2314;0.9961 0.7569 0.2275;0.9961 0.7608 0.2235;0.9961 0.7686 0.2196;0.9961 0.7725 0.2157;0.9961 0.7804 0.2078;0.9961 0.7843 0.2039;0.9961 0.7922 0.2;0.9922 0.7961 0.1961;0.9922 0.8039 0.1922;0.9922 0.8078 0.1922;0.9882 0.8157 0.1882;0.9843 0.8235 0.1843;0.9843 0.8275 0.1804;0.9804 0.8353 0.1804;0.9765 0.8392 0.1765;0.9765 0.8471 0.1725;0.9725 0.851 0.1686;0.9686 0.8588 0.1647;0.9686 0.8667 0.1647;0.9647 0.8706 0.1608;0.9647 0.8784 0.1569;0.9608 0.8824 0.1569;0.9608 0.8902 0.1529;0.9608 0.898 0.149;0.9608 0.902 0.149;0.9608 0.9098 0.1451;0.9608 0.9137 0.1412;0.9608 0.9216 0.1373;0.9608 0.9255 0.1333;0.9608 0.9333 0.1294;0.9647 0.9373 0.1255;0.9647 0.9451 0.1216;0.9647 0.949 0.1176;0.9686 0.9569 0.1098;0.9686 0.9608 0.1059;0.9725 0.9686 0.102;0.9725 0.9725 0.0941;0.9765 0.9765 0.0863;0.9765 0.9843 0.0824];
                app.g6.FontSize = 11.3333333333333;
                app.g6.NextPlot = 'replace';
                app.g6.Tag = 'g6';
                app.g6.Position = [440 185 273 200];
                
                % Create g7 小波变换4
                app.g7 = uiaxes(app.uibuttongroup11);
                app.g7.FontName = 'Microsoft YaHei UI';
                app.g7.Colormap = [0.2431 0.149 0.6588;0.2431 0.1529 0.6745;0.2471 0.1569 0.6863;0.2471 0.1608 0.698;0.251 0.1647 0.7059;0.251 0.1686 0.7176;0.2549 0.1725 0.7294;0.2549 0.1765 0.7412;0.2588 0.1804 0.749;0.2588 0.1843 0.7608;0.2627 0.1922 0.7843;0.2627 0.1961 0.7922;0.2667 0.2 0.8039;0.2667 0.2039 0.8157;0.2706 0.2078 0.8235;0.2706 0.2157 0.8353;0.2706 0.2196 0.8431;0.2745 0.2235 0.851;0.2745 0.2275 0.8627;0.2745 0.2314 0.8706;0.2745 0.2392 0.8784;0.2784 0.2431 0.8824;0.2784 0.2471 0.8902;0.2784 0.2549 0.898;0.2784 0.2588 0.902;0.2784 0.2667 0.9098;0.2784 0.2706 0.9137;0.2784 0.2745 0.9216;0.2824 0.2824 0.9255;0.2824 0.2863 0.9294;0.2824 0.2941 0.9333;0.2824 0.298 0.9412;0.2824 0.3059 0.9451;0.2824 0.3098 0.949;0.2824 0.3137 0.9529;0.2824 0.3216 0.9569;0.2824 0.3255 0.9608;0.2824 0.3294 0.9647;0.2784 0.3373 0.9686;0.2784 0.3412 0.9686;0.2784 0.349 0.9725;0.2784 0.3529 0.9765;0.2784 0.3569 0.9804;0.2784 0.3647 0.9804;0.2745 0.3686 0.9843;0.2745 0.3765 0.9843;0.2745 0.3804 0.9882;0.2706 0.3843 0.9882;0.2706 0.3922 0.9922;0.2667 0.3961 0.9922;0.2627 0.4039 0.9922;0.2627 0.4078 0.9961;0.2588 0.4157 0.9961;0.2549 0.4196 0.9961;0.251 0.4275 0.9961;0.2471 0.4314 1;0.2431 0.4392 1;0.2353 0.4431 1;0.2314 0.451 1;0.2235 0.4549 1;0.2196 0.4627 0.9961;0.2118 0.4667 0.9961;0.2078 0.4745 0.9922;0.2 0.4784 0.9922;0.1961 0.4863 0.9882;0.1922 0.4902 0.9882;0.1882 0.498 0.9843;0.1843 0.502 0.9804;0.1843 0.5098 0.9804;0.1804 0.5137 0.9765;0.1804 0.5176 0.9725;0.1804 0.5255 0.9725;0.1804 0.5294 0.9686;0.1765 0.5333 0.9647;0.1765 0.5412 0.9608;0.1765 0.5451 0.9569;0.1765 0.549 0.9529;0.1765 0.5569 0.949;0.1725 0.5608 0.9451;0.1725 0.5647 0.9412;0.1686 0.5686 0.9373;0.1647 0.5765 0.9333;0.1608 0.5804 0.9294;0.1569 0.5843 0.9255;0.1529 0.5922 0.9216;0.1529 0.5961 0.9176;0.149 0.6 0.9137;0.149 0.6039 0.9098;0.1451 0.6078 0.9098;0.1451 0.6118 0.9059;0.1412 0.6196 0.902;0.1412 0.6235 0.898;0.1373 0.6275 0.898;0.1373 0.6314 0.8941;0.1333 0.6353 0.8941;0.1294 0.6392 0.8902;0.1255 0.6471 0.8902;0.1216 0.651 0.8863;0.1176 0.6549 0.8824;0.1137 0.6588 0.8824;0.1137 0.6627 0.8784;0.1098 0.6667 0.8745;0.1059 0.6706 0.8706;0.102 0.6745 0.8667;0.098 0.6784 0.8627;0.0902 0.6824 0.8549;0.0863 0.6863 0.851;0.0784 0.6902 0.8471;0.0706 0.6941 0.8392;0.0627 0.698 0.8353;0.0549 0.702 0.8314;0.0431 0.702 0.8235;0.0314 0.7059 0.8196;0.0235 0.7098 0.8118;0.0157 0.7137 0.8078;0.0078 0.7176 0.8;0.0039 0.7176 0.7922;0 0.7216 0.7882;0 0.7255 0.7804;0 0.7294 0.7765;0.0039 0.7294 0.7686;0.0078 0.7333 0.7608;0.0157 0.7333 0.7569;0.0235 0.7373 0.749;0.0353 0.7412 0.7412;0.051 0.7412 0.7373;0.0627 0.7451 0.7294;0.0784 0.7451 0.7216;0.0902 0.749 0.7137;0.102 0.7529 0.7098;0.1137 0.7529 0.702;0.1255 0.7569 0.6941;0.1373 0.7569 0.6863;0.1451 0.7608 0.6824;0.1529 0.7608 0.6745;0.1608 0.7647 0.6667;0.1686 0.7647 0.6588;0.1725 0.7686 0.651;0.1804 0.7686 0.6471;0.1843 0.7725 0.6392;0.1922 0.7725 0.6314;0.1961 0.7765 0.6235;0.2 0.7804 0.6157;0.2078 0.7804 0.6078;0.2118 0.7843 0.6;0.2196 0.7843 0.5882;0.2235 0.7882 0.5804;0.2314 0.7882 0.5725;0.2392 0.7922 0.5647;0.251 0.7922 0.5529;0.2588 0.7922 0.5451;0.2706 0.7961 0.5373;0.2824 0.7961 0.5255;0.2941 0.7961 0.5176;0.3059 0.8 0.5059;0.3176 0.8 0.498;0.3294 0.8 0.4863;0.3412 0.8 0.4784;0.3529 0.8 0.4667;0.3686 0.8039 0.4549;0.3804 0.8039 0.4471;0.3922 0.8039 0.4353;0.4039 0.8039 0.4235;0.4196 0.8039 0.4118;0.4314 0.8039 0.4;0.4471 0.8039 0.3922;0.4627 0.8 0.3804;0.4745 0.8 0.3686;0.4902 0.8 0.3569;0.5059 0.8 0.349;0.5176 0.8 0.3373;0.5333 0.7961 0.3255;0.5451 0.7961 0.3176;0.5608 0.7961 0.3059;0.5765 0.7922 0.2941;0.5882 0.7922 0.2824;0.6039 0.7882 0.2745;0.6157 0.7882 0.2627;0.6314 0.7843 0.251;0.6431 0.7843 0.2431;0.6549 0.7804 0.2314;0.6706 0.7804 0.2235;0.6824 0.7765 0.2157;0.698 0.7765 0.2078;0.7098 0.7725 0.2;0.7216 0.7686 0.1922;0.7333 0.7686 0.1843;0.7451 0.7647 0.1765;0.7608 0.7647 0.1725;0.7725 0.7608 0.1647;0.7843 0.7569 0.1608;0.7961 0.7569 0.1569;0.8078 0.7529 0.1529;0.8157 0.749 0.1529;0.8275 0.749 0.1529;0.8392 0.7451 0.1529;0.851 0.7451 0.1569;0.8588 0.7412 0.1569;0.8706 0.7373 0.1608;0.8824 0.7373 0.1647;0.8902 0.7373 0.1686;0.902 0.7333 0.1765;0.9098 0.7333 0.1804;0.9176 0.7294 0.1882;0.9255 0.7294 0.1961;0.9373 0.7294 0.2078;0.9451 0.7294 0.2157;0.9529 0.7294 0.2235;0.9608 0.7294 0.2314;0.9686 0.7294 0.2392;0.9765 0.7294 0.2431;0.9843 0.7333 0.2431;0.9882 0.7373 0.2431;0.9961 0.7412 0.2392;0.9961 0.7451 0.2353;0.9961 0.7529 0.2314;0.9961 0.7569 0.2275;0.9961 0.7608 0.2235;0.9961 0.7686 0.2196;0.9961 0.7725 0.2157;0.9961 0.7804 0.2078;0.9961 0.7843 0.2039;0.9961 0.7922 0.2;0.9922 0.7961 0.1961;0.9922 0.8039 0.1922;0.9922 0.8078 0.1922;0.9882 0.8157 0.1882;0.9843 0.8235 0.1843;0.9843 0.8275 0.1804;0.9804 0.8353 0.1804;0.9765 0.8392 0.1765;0.9765 0.8471 0.1725;0.9725 0.851 0.1686;0.9686 0.8588 0.1647;0.9686 0.8667 0.1647;0.9647 0.8706 0.1608;0.9647 0.8784 0.1569;0.9608 0.8824 0.1569;0.9608 0.8902 0.1529;0.9608 0.898 0.149;0.9608 0.902 0.149;0.9608 0.9098 0.1451;0.9608 0.9137 0.1412;0.9608 0.9216 0.1373;0.9608 0.9255 0.1333;0.9608 0.9333 0.1294;0.9647 0.9373 0.1255;0.9647 0.9451 0.1216;0.9647 0.949 0.1176;0.9686 0.9569 0.1098;0.9686 0.9608 0.1059;0.9725 0.9686 0.102;0.9725 0.9725 0.0941;0.9765 0.9765 0.0863;0.9765 0.9843 0.0824];
                app.g7.FontSize = 11.3333333333333;
                app.g7.NextPlot = 'replace';
                app.g7.Tag = 'g7';
                app.g7.Position = [440 0 273 200];
                
                % Create uibuttongroup12_1
                app.uibuttongroup12_1 = uibuttongroup(app.uibuttongroup11);
                app.uibuttongroup12_1.Title = '小波选择';
                app.uibuttongroup12_1.Tag = 'uibuttongroup12_1';
                app.uibuttongroup12_1.FontName = 'Microsoft YaHei UI';
                app.uibuttongroup12_1.FontSize = 16;
                app.uibuttongroup12_1.Position = [350 480 133 260];
                
                % Create Label
                app.Label = uilabel(app.uibuttongroup12_1);
                app.Label.FontWeight = 'bold';
                app.Label.FontAngle = 'italic';
                app.Label.FontColor = [0.149 0.149 0.149];
                app.Label.Position = [20 0 150 20];
                app.Label.Text = '（默认为harr小波）';
                
                % Create Button1
                app.Button1 = uibutton(app.uibuttongroup12_1, 'push');
                app.Button1.ButtonPushedFcn = createCallbackFcn(app, @Button1Pushed, true);
                app.Button1.Position = [19 200 100 22];
                app.Button1.Tag = 'Button1';
                app.Button1.Text = 'haar';
                
                % Create Button2
                app.Button2 = uibutton(app.uibuttongroup12_1, 'push');
                app.Button2.ButtonPushedFcn = createCallbackFcn(app, @Button2Pushed, true);
                app.Button2.Position = [19 170 100 22];
                app.Button2.Tag = 'Button2';
                app.Button2.Text = 'db2';
                
                % Create Button3
                app.Button3 = uibutton(app.uibuttongroup12_1, 'push');
                app.Button3.ButtonPushedFcn = createCallbackFcn(app, @Button3Pushed, true);
                app.Button3.Position = [19 140 100 22];
                app.Button3.Tag = 'Button3';
                app.Button3.Text = 'bior1.1';
                
                % Create Button4
                app.Button4 = uibutton(app.uibuttongroup12_1, 'push');
                app.Button4.ButtonPushedFcn = createCallbackFcn(app, @Button4Pushed, true);
                app.Button4.Position = [19 110 100 22];
                app.Button4.Tag = 'Button4';
                app.Button4.Text = 'coif1';
                
                % Create Button5
                app.Button5 = uibutton(app.uibuttongroup12_1, 'push');
                app.Button5.ButtonPushedFcn = createCallbackFcn(app, @Button5Pushed, true);
                app.Button5.Position = [19 80 100 22];
                app.Button5.Tag = 'Button5';
                app.Button5.Text = 'sym2';
                
                % Create Button6
                app.Button6 = uibutton(app.uibuttongroup12_1, 'push');
                app.Button6.ButtonPushedFcn = createCallbackFcn(app, @Button6Pushed, true);
                app.Button6.Position = [19 50 100 22];
                app.Button6.Tag = 'Button6';
                app.Button6.Text = 'fk4';
                
                % Create Button7
                app.Button7 = uibutton(app.uibuttongroup12_1, 'push');
                app.Button7.ButtonPushedFcn = createCallbackFcn(app, @Button7Pushed, true);
                app.Button7.Position = [19 20 100 22];
                app.Button7.Tag = 'Button7';
                app.Button7.Text = 'dmey';
                
            catch
                errordlg('发生了错误，请检查输入或操作','错误','modal');
            end
            
            % Show the figure after all components are created
            app.figure.Visible = 'on';
        end
    end
    
    % UI界面生成与删除
    methods (Access = public)
        
        % Construct app
        function app = Image_Processing_GUI(varargin)
            
            runningApp = getRunningApp(app);
            
            % 检查是否存在正在运行的单例App
            if isempty(runningApp)
                
                try
                    % 创建UIFigure和组件
                    createComponents(app)
                catch
                    errordlg('发生了错误，请检查输入或操作','错误','modal');
                end
                % 将App注册到App Designer
                registerApp(app, app.figure)
                
                % 执行启动函数
                runStartupFcn(app, @(app)Image_processing_GUI_OpeningFcn(app, varargin{:}))
                
            else
                % 聚焦正在运行的单例App
                figure(runningApp.figure)
                
                app = runningApp;
            end
            
            if nargout == 0
                clear app
            end
        end
        
        % Code that executes before app deletion
        function delete(app)
            % 在app被删除时删除UIFigure
            delete(app.figure)
        end
    end
end