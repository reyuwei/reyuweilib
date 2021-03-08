%% ---Just 2018-6-7 -----
% ***************************

clc;clear;
% folders_path = 'X:\Just\MyProjects\POSE\data\test\Calib1226';
% save_file = 'X:\Just\MyProjects\POSE\data\test\outputxml\volscan.xml';
image_size = [960,600];
% folders_path =  'D:\Just\POSE\_DataForSiggraph_POSE\tools\OurMVS\Data\Calib\Calib_0513_multireso';
% save_file = 'D:\Just\POSE\_DataForSiggraph_POSE\tools\OurMVS\Data\Calib\test_sportman_agi.xml';
% image_size = [960,600]; % width - height
folders_path = 'C:\Users\liyuwei\Desktop\real\girl_1_rotated\calib_folder';
save_file = 'C:\Users\liyuwei\Desktop\real\girl_1_rotated\calib_folder\agi.xml';
% folders_path = 'L:\data\jiewu-skr\calib-0106';
% save_file = 'L:\data\jiewu-skr\calib-0106.xml';

multi_reso = [];

%% conform resolution info,
Idmap_path = [folders_path '\Idmap.txt'];
fileID = fopen(Idmap_path);
data = textscan(fileID,'%s %s %s %s');
cam_num = str2double(cell2mat(data{1,1}(1)));
if strcmp(cell2mat(data{1,3}(1)),'"Resolution"')
    for i=1:cam_num
        multi_reso = [multi_reso [data{1,3}(i+1),data{1,4}(i+1)]];
    end
else
    for i=1:cam_num
        multi_reso = [multi_reso {num2str(image_size(1)),num2str(image_size(2))}];
    end
end
calib_folder_names = [];
for i=1:cam_num
    name = data{1,2}(i+1);
    %     name =  name{1,1}(2:end-1);
    calib_folder_names = [calib_folder_names,name];
end
fclose(fileID);

%% read all xml files
extrinsics=[];
intrinsics=[];
for i = 1:length(calib_folder_names)
    name = calib_folder_names{i}(2:end-1);
    path = fullfile(folders_path,name);
    xml_paths = dir(strcat(path,'\*.xml'));
    for j = 1:length(xml_paths)
        xml_path = fullfile(xml_paths(j).folder, xml_paths(j).name);
        xMat = xml2struct(xml_path);
        if strcmp(xml_paths(j).name,'extrinsics.xml')
            extrinsics = [extrinsics xMat];
        else
            intrinsics = [intrinsics xMat];
        end
    end
    
end

disp('Read .xml files successfully !' );

%% write to the final xml
docNode = com.mathworks.xml.XMLUtils.createDocument('document');
document = docNode.getDocumentElement;
document.setAttribute('version','0.9.1');

chunkElement = docNode.createElement('chunk');
document.appendChild(chunkElement);

% write intrinsics properties
sensorElement = docNode.createElement('sensors');
for i=1:size(intrinsics,2)
    % attribute
    thisElement = docNode.createElement('sensor');
    thisElement.setAttribute('id',num2str(i-1));
    thisElement.setAttribute('label','unknown');
    thisElement.setAttribute('type','frame');
    % resolution
    resolElement = docNode.createElement('resolution');
    resolElement.setAttribute('width',multi_reso(2*i-1));
    resolElement.setAttribute('height',multi_reso(2*i));
    thisElement.appendChild(resolElement);
    % property

%     propElement = docNode.createElement('property');
%     propElement.setAttribute('name','pixel_width');
%     propElement.setAttribute('value','5.9815939663530058e-003');
%     thisElement.appendChild(propElement);
%     
%     propElement = docNode.createElement('property');
%     propElement.setAttribute('name','pixel_height');
%     propElement.setAttribute('value','5.9815939663530058e-003');
%     thisElement.appendChild(propElement);
    
    %propElement = docNode.createElement('property');
    %propElement.setAttribute('name','focal_length');
    %propElement.setAttribute('value','6.6000000000000000e+001');
    %thisElement.appendChild(propElement);
    
    propElement = docNode.createElement('property');
    propElement.setAttribute('name','fixed');
    propElement.setAttribute('value','false');
    thisElement.appendChild(propElement);
    
    
    % calibration
    calibElement = docNode.createElement('calibration');
    calibElement.setAttribute('type','frame');
    calibElement.setAttribute('class','adjusted');
    resolElement2 = docNode.createElement('resolution');
    resolElement2.setAttribute('width',multi_reso(2*i-1));
    resolElement2.setAttribute('height',multi_reso(2*i));
    calibElement.appendChild(resolElement2);
    
    M_char = intrinsics(i).opencv_storage.M.data.Text;
    D_char = intrinsics(i).opencv_storage.D.data.Text;
    
    M_char = strtrim(M_char);
    M_data = regexp(M_char,' ','split');
    M_data(cellfun(@isempty,M_data)) = [];
    D_char = strtrim(D_char);
    D_data = regexp(D_char,' ','split');
    D_data(cellfun(@isempty,D_data)) = [];
    
    inElement = docNode.createElement('fx');
    inElement.appendChild(docNode.createTextNode(M_data{1}));
    calibElement.appendChild(inElement);
    
    inElement = docNode.createElement('fy');
    inElement.appendChild(docNode.createTextNode(M_data{5}));
    calibElement.appendChild(inElement);
    
    inElement = docNode.createElement('cx');
    inElement.appendChild(docNode.createTextNode(M_data{3}));
    calibElement.appendChild(inElement);
    
    inElement = docNode.createElement('cy');
    inElement.appendChild(docNode.createTextNode(M_data{6}));
    calibElement.appendChild(inElement);
    
    %inElement = docNode.createElement('skew');
    %inElement.appendChild(docNode.createTextNode(M_data{2}));
    %calibElement.appendChild(inElement);
    
    inElement = docNode.createElement('k1');
    inElement.appendChild(docNode.createTextNode(D_data{1}));
    calibElement.appendChild(inElement);
    
    inElement = docNode.createElement('k2');
    inElement.appendChild(docNode.createTextNode(D_data{2}));
    calibElement.appendChild(inElement);
    
    % !!
    inElement = docNode.createElement('k3');
    inElement.appendChild(docNode.createTextNode(D_data{5}));
    calibElement.appendChild(inElement);
    
    %inElement = docNode.createElement('p1');
    %inElement.appendChild(docNode.createTextNode(D_data{3}));
    %calibElement.appendChild(inElement);
    
    %inElement = docNode.createElement('p2');
    %inElement.appendChild(docNode.createTextNode(D_data{4}));
    %calibElement.appendChild(inElement);
    
    thisElement.appendChild(calibElement);
    sensorElement.appendChild(thisElement);
end
chunkElement.appendChild(sensorElement);


% write extrinsics properties
camerasElement = docNode.createElement('cameras');
for i = 1:size(extrinsics, 2)
    % attribute
    thisElement = docNode.createElement('camera');
    thisElement.setAttribute('id',num2str(i-1));
    thisElement.setAttribute('label', ['image.cam' num2str(i,'%02d') '_000000.png']);
    thisElement.setAttribute('sensor_id',num2str(i-1));
    thisElement.setAttribute('enabled','true');
    
    % resolution
    resolElement = docNode.createElement('resolution');
    resolElement.setAttribute('width',multi_reso(2*i-1));
    resolElement.setAttribute('height',multi_reso(2*i));
    thisElement.appendChild(resolElement);
    
    
    % transform
    transElement = docNode.createElement('transform');
    R_char = extrinsics(i).opencv_storage.R.data.Text;
    R_char = strtrim(R_char);
    T_char = extrinsics(i).opencv_storage.T.data.Text;
    T_char = strtrim(T_char);
    R_data = regexp(R_char,' ','split');
    R_data(cellfun(@isempty,R_data)) = [];
    T_data = regexp(T_char,' ','split');
    T_data(cellfun(@isempty,T_data)) = [];
    
    RT_mat = zeros(4,4);
    RT_mat(1) = str2double(cell2mat(R_data(1)));
    RT_mat(2) = str2double(cell2mat(R_data(2)));
    RT_mat(3) = str2double(cell2mat(R_data(3)));
    RT_mat(5) = str2double(cell2mat(R_data(4)));
    RT_mat(6) = str2double(cell2mat(R_data(5)));
    RT_mat(7) = str2double(cell2mat(R_data(6)));
    RT_mat(9) = str2double(cell2mat(R_data(7)));
    RT_mat(10) = str2double(cell2mat(R_data(8)));
    RT_mat(11) = str2double(cell2mat(R_data(9)));
    RT_mat(13) = str2double(cell2mat(T_data(1)));
    RT_mat(14) = str2double(cell2mat(T_data(2)));
    RT_mat(15) = str2double(cell2mat(T_data(3)));
    RT_mat(16) = 1.0;
    
    % T = -R*T !!
    RT_mat(1:3, 4) = -RT_mat(1:3, 1:3) * RT_mat(1:3, 4);
    
    my_trans_char = mat2str(RT_mat,16);
    my_trans_char = strrep(my_trans_char,';',' ');
    my_trans_char = strip(my_trans_char,'left','[');
    my_trans_char = strip(my_trans_char,'right',']');
    transElement.appendChild(docNode.createTextNode(my_trans_char))
    thisElement.appendChild(transElement);
    
    camerasElement.appendChild(thisElement);
end
chunkElement.appendChild(camerasElement);

% setting
settingElement = docNode.createElement('settings');
chunkElement.appendChild(settingElement);

propElement = docNode.createElement('property');
propElement.setAttribute('name','accuracy_tiepoints');
propElement.setAttribute('value','1');
settingElement.appendChild(propElement);

propElement = docNode.createElement('property');
propElement.setAttribute('name','accuracy_cameras');
propElement.setAttribute('value','10');
settingElement.appendChild(propElement);

propElement = docNode.createElement('property');
propElement.setAttribute('name','accuracy_cameras_ypr');
propElement.setAttribute('value','2');
settingElement.appendChild(propElement);

propElement = docNode.createElement('property');
propElement.setAttribute('name','accuracy_markers');
propElement.setAttribute('value','0.0050000000000000001');
settingElement.appendChild(propElement);

propElement = docNode.createElement('property');
propElement.setAttribute('name','accuracy_scalebars');
propElement.setAttribute('value','0.001');
settingElement.appendChild(propElement);

propElement = docNode.createElement('property');
propElement.setAttribute('name','accuracy_projections');
propElement.setAttribute('value','0.10000000000000001');
settingElement.appendChild(propElement);

xmlFileName = save_file;
xmlwrite(xmlFileName,docNode);
type(xmlFileName);

disp('Convert to agi.xml successfully !' );