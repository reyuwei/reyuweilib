clc;clear;
save_dir = 'C:\Users\liyuwei\Desktop\calib_1121\calibfolder_1121';
xml_path = 'C:\Users\liyuwei\Desktop\calib_1121\calib1121.xml';
xDoc  = xml2struct(xml_path);

mkdir(save_dir);

intrXml = xDoc.document.chunk.sensors.sensor;
extrXml = xDoc.document.chunk.cameras.camera;
 
Intrs = {};
Extrs = {};
  
for i = 1:length(intrXml)
    
    intr = struct('w',0,'h',0,'fx', 0, 'fy', 0, 'cx', 0, 'cy', 0, 'skew', 0, 'k1', 0, 'k2', 0, 'k3', 0, 'p1', 0, 'p2', 0, 'label', ' ', 'sensor', 0);
    
    if(~isfield(intrXml{i}, 'calibration'))
        continue ;
    end
    
    intr.w = intrXml{i}.calibration.resolution.Attributes.width;
    intr.h = intrXml{i}.calibration.resolution.Attributes.height;
    intr.fx = cell2mat(textscan(intrXml{i}.calibration.fx.Text,'%f64'));
    intr.fy = cell2mat(textscan(intrXml{i}.calibration.fy.Text,'%f64'));
    intr.cx = cell2mat(textscan(intrXml{i}.calibration.cx.Text,'%f64'));
    intr.cy = cell2mat(textscan(intrXml{i}.calibration.cy.Text,'%f64'));

    
    if(isfield(intrXml{i}.calibration, 'skew'))
       intr.skew = cell2mat(textscan(intrXml{i}.calibration.skew.Text,'%f64'));
    end
    %intr.skew = cell2mat(textscan(intrXml{i}.calibration.skew.Text,'%f64'));
    
    if(isfield(intrXml{i}.calibration, 'k1'))
        intr.k1 = cell2mat(textscan(intrXml{i}.calibration.k1.Text,'%f64'));
    end
    
    if(isfield(intrXml{i}.calibration, 'k2'))
        intr.k2 = cell2mat(textscan(intrXml{i}.calibration.k2.Text,'%f64'));
    end
    
    if(isfield(intrXml{i}.calibration, 'k3'))
        intr.k3 = cell2mat(textscan(intrXml{i}.calibration.k3.Text,'%f64'));
    end
    
    if(isfield(intrXml{i}.calibration, 'p1'))
        intr.p1 = cell2mat(textscan(intrXml{i}.calibration.p1.Text,'%f64'));
    end
    
    if(isfield(intrXml{i}.calibration, 'p2'))
        intr.p2 = cell2mat(textscan(intrXml{i}.calibration.p2.Text,'%f64'));
    end
    
    intr.label = intrXml{i}.Attributes.label;
    intr.sensor = cell2mat(textscan(intrXml{i}.Attributes.id, '%d'));
    Intrs = [Intrs, {intr}];
end

for i = 1:length(extrXml)
   
    extr = struct('M', 0, 'label', ' ', 'sensor', 0);
    
    RTm = cell2mat(textscan(extrXml{i}.transform.Text,'%f64'));
    extr.M = reshape(RTm, [4, 4])';
    extr.label = extrXml{i}.Attributes.label;
    extr.sensor = cell2mat(textscan(extrXml{i}.Attributes.sensor_id, '%d'));
    Extrs = [Extrs, {extr}];
end

if( length(Intrs) ~= length(Extrs) )
   disp('WARNING: Parsed Number of Intrinsics and Extrinsics Does not Match!'); 
end

Cams = {};
for i = 1:length(Extrs)
    cam = struct('Intr', 0, 'Extr', 0, 'label', ' ', 'camid', 0, 'id', 0, 'sensor', 0);
    cam.Extr = Extrs{i};

    cam.label = Extrs{i}.label;
    cam_frame = textscan(cam.label, 'img.%u64_%d.png');
    cam.camid = cam_frame{1};
    cam.id = i - 1;
    cam.sensor = Extrs{i}.sensor;
    
    found = false;
    for j = 1:length(Intrs)
        if( Intrs{j}.sensor == Extrs{i}.sensor)
            cam.Intr = Intrs{j};
            found = true;
            break
        end
    end
    
    if(~found)
        disp('Intrinsics and Extrinsics Labels Do not Match!'); 
    	return 
    end
    
    Cams = [Cams {cam}];
end

idfilename = sprintf('%s/Idmap.txt', save_dir);
idfileID = fopen(idfilename,'w');
fprintf(idfileID, '%d %d "Resolution"\n', length(Cams), length(Cams));

for i = 1:length(Cams)
    %str1 = strsplit(Cams{i}.label,'_');
    %str2 = strsplit(str1{3},'.');
    str2 = strsplit(Cams{i}.label,'.');
    str = str2(1);
    
    intr = Cams{i}.Intr;
    extr = Cams{i}.Extr;
    
    mkdir(save_dir, str{1});
     
    K  = [intr.fx, intr.skew, intr.cx; 0, intr.fy, intr.cy;0, 0, 1];
    RT = extr.M(1:3, :);
    RT(1:3, 4) = -inv(RT(1:3, 1:3)) * RT(1:3, 4);
    RT(1:3, 1:3) = inv(RT(1:3, 1:3));
    
    R = RT(1:3, 1:3);
    T = RT(1:3, 4);
    
    P = K * RT;
    D = [intr.k1, intr.k2,  intr.p1, intr.p2,intr.k3];
    
    a = str2num(Cams{i}.Intr.w);
    b = str2num(Cams{i}.Intr.h);
    fprintf(idfileID, '%d %s %d %d\n', i, ['"' str{1} '"'], a, b);
    
    filename = sprintf('%s/%s/intrinsic.xml', save_dir, str{1});
    fileID = fopen(filename,'w');
    
    fprintf(fileID,'<?xml version="1.0"?>\n');
    fprintf(fileID,'<opencv_storage>\n');
    fprintf(fileID,'<date>Calib0527-2147</date>\n');
    fprintf(fileID,'<Project_error>1.00e-02</Project_error>\n');
    
    fprintf(fileID,'<M type_id="opencv-matrix">\n');
    fprintf(fileID,'  <rows>3</rows>\n');
    fprintf(fileID,'  <cols>3</cols>\n');
    fprintf(fileID,'  <dt>d</dt>\n');
    fprintf(fileID,'  <data>\n');
    fprintf(fileID,'    %f %f %f %f\n', K(1, 1), K(1, 2), K(1, 3), K(2, 1));
    fprintf(fileID,'    %f %f %f %f %f</data></M>\n', K(2, 2), K(2, 3), K(3, 1), K(3, 2), K(3, 3));
    
    fprintf(fileID,'<D type_id="opencv-matrix">\n');
    fprintf(fileID,'  <rows>1</rows>\n');
    fprintf(fileID,'  <cols>5</cols>\n');
    fprintf(fileID,'  <dt>d</dt>\n');
    fprintf(fileID,'  <data>\n');
    fprintf(fileID,'    %f %f\n', D(1), D(2));
    fprintf(fileID,'    %f %f\n', D(3), D(4));
    fprintf(fileID,'    %f</data></D>\n', D(5));
    fprintf(fileID,'</opencv_storage>\n');
    
    fclose(fileID);
    
    filename = sprintf('%s/%s/extrinsics.xml', save_dir, str{1});
    fileID = fopen(filename,'w');
    
    fprintf(fileID,'<?xml version="1.0"?>\n');
    fprintf(fileID,'<opencv_storage>\n');
    
    fprintf(fileID,'<R type_id="opencv-matrix">\n');
    fprintf(fileID,'  <rows>3</rows>\n');
    fprintf(fileID,'  <cols>3</cols>\n');
    fprintf(fileID,'  <dt>d</dt>\n');
    fprintf(fileID,'  <data>\n');
    fprintf(fileID,'    %f %f\n', R(1, 1), R(1, 2));
    fprintf(fileID,'    %f %f\n', R(1, 3), R(2, 1));
    fprintf(fileID,'    %f %f %f\n', R(2, 2), R(2, 3), R(3, 1));
    fprintf(fileID,'    %f %f</data></R>\n', R(3, 2), R(3, 3));
    
    fprintf(fileID,'<T type_id="opencv-matrix">\n');
    fprintf(fileID,'  <rows>3</rows>\n');
    fprintf(fileID,'  <cols>1</cols>\n');
    fprintf(fileID,'  <dt>d</dt>\n');
    fprintf(fileID,'  <data>\n');
    fprintf(fileID,'    %f %f %f</data></T>\n', T(1), T(2), T(3));
    fprintf(fileID,'</opencv_storage>\n');
    
    fclose(fileID);
        
%     wR = RT(1:3, 1:3);
%     wT = RT(1:3, 4);
    wR = inv(RT(1:3, 1:3));
    wT = -inv(RT(1:3, 1:3)) * RT(1:3, 4);
    plotcamera('Transform', [wR wT], 'Thickness', 2, 'Fill', 1, 'Scale', [0.2 0.2 0.2], 'Righthand', 1, 'Color', [0 0 1]);
    hold on
    text(wT(1), wT(2), wT(3),Cams{i}.label)
    
end

fclose(idfileID);

configfilename = sprintf('%s/config_gpu.xml', save_dir);
configfileID = fopen(configfilename,'w');
fprintf(configfileID, '<?xml version="1.0"?>\n');
fprintf(configfileID, '<opencv_storage>\n');
fprintf(configfileID, '<frame_seq_format>image.cam%%02d_%%05d.png</frame_seq_format>\n');
fprintf(configfileID, '<mask_seq_format>mask/image.cam%%02d_%%05d.png</mask_seq_format>\n');
fprintf(configfileID, '<calibration_file_format>INDIVIDUAL</calibration_file_format>\n');
fprintf(configfileID, '<calibration_file_format_types>INDIVIDUAL DREWS1.0</calibration_file_format_types>\n');
fprintf(configfileID, '<calibration_folder>new_calibration/calibration</calibration_folder>\n');
fprintf(configfileID, '<calibration_file_name>synth_0.out</calibration_file_name>\n');
fprintf(configfileID, '<result_folder>result_gpu</result_folder>\n');
fprintf(configfileID, '<computation_step>0</computation_step>\n');
fprintf(configfileID, '<skip_computation>0</skip_computation>\n');
fprintf(configfileID, '<output_camera_params>PMVS</output_camera_params>\n');
fprintf(configfileID, '<output_camera_params_types>ORIGINAL UNDISTORTED STEREO PMVS AGISOFT</output_camera_params_types>\n');
fprintf(configfileID, '<output_camera_params_name>full_cameras.out</output_camera_params_name>\n');
fprintf(configfileID, '<frame_id_base>0</frame_id_base>\n');
fprintf(configfileID, '<frame_num>1</frame_num>\n');
fprintf(configfileID, '<cam_id_base>1</cam_id_base>\n');
fprintf(configfileID, '<cam_id_cap>%d</cam_id_cap>\n', length(Cams));
fprintf(configfileID, '<distance_obj_to_cam>3.8</distance_obj_to_cam>\n');
fprintf(configfileID, '<disparity_size>512</disparity_size>\n');
fprintf(configfileID, '<down_scale>1.0</down_scale>\n');
fprintf(configfileID, '<frame_width>1000</frame_width>\n');
fprintf(configfileID, '<frame_height>1500</frame_height>\n');
fprintf(configfileID, '<roi_width>1000</roi_width>\n');
fprintf(configfileID, '<roi_height>1500</roi_height>\n');
fprintf(configfileID, '<camera_max_id>%d</camera_max_id>\n', length(Cams));
fprintf(configfileID, '<debug_output>0</debug_output>\n');


for i = 1:length(Cams)

    fprintf(configfileID, '<reference_cam_%d>\n', i);
    fprintf(configfileID, '\t<other_id_list>%d</other_id_list>\n', i);
    fprintf(configfileID, '</reference_cam_%d>\n', i);
    
end

fprintf(configfileID, '</opencv_storage>\n');

fclose(configfileID);
