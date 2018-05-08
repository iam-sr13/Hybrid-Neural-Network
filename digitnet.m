%% Load data

digitds = imageDatastore('D:\Codebase\OctaveMatlabSpace\BNN\mnistasjpg\trainingSet','IncludeSubfolders',true,'LabelSource','foldernames','ReadFcn',@digitsread);
[trainImgs,testImgs] = splitEachLabel(digitds,0.8);
[trainsnn,testsnn] = splitEachLabel(digitds,0.8);
numClasses = numel(categories(digitds.Labels));

net = alexnet;
layers = net.Layers;
layers(end-2) = fullyConnectedLayer(numClasses);
layers(end) = classificationLayer;
options = trainingOptions('sgdm','InitialLearnRate', 0.001);

%% Train Alexnet
[digitNet,info] = trainNetwork(trainImgs, layers, options);
testpreds = classify(digitNet,testImgs);

nnz(testpreds == testImgs.Labels)/numel(testpreds)
[digitsconf,digitsnames] = confusionmat(testImgs.Labels,testpreds);
heatmap(digitsnames,digitsnames,digitsconf)

im = readimage(testImgs,1);
imshow(im)
act1 = activations(digitNet,im,'conv1','OutputAs','channels');
sz = size(act1);
act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
montage(mat2gray(act1),'Size',[8 12])
testpreds = classify(digitNet,im)


%% Use Hybrid Transfer learning on SNN from DNN
%reset(trainImgs)
%hasdata(trainImgs)

train_x=zeros(33601,9216);
train_y=zeros(33601,10);
test_x=zeros(0,9216);
test_y=zeros(0,10);

count=0;
fprintf('Beginning Hybrid Transfer Learning... \n Step 1: Transferring Intermediate Knowledge from DNN to SNN...\n');

while hasdata(trainsnn)
    [data,info] = read(trainsnn);    
    %grim=rgb2gray(data);
    actalex=activations(digitNet,data,'pool5','OutputAs','rows');
    train_x(count+1,:)= actalex;
    train_y(count+1,:)= string2label(char(info.Label));
    count=count+1;
    fprintf('%.2f%% complete...\n',count/33601*100);
end

count=0;
fprintf('Step 2: Preparing SNN to train on the acquired knowledge...\n');

while hasdata(testsnn)
    [data,info] = read(testsnn);
    %grim=rgb2gray(data);
    actalex=activations(digitNet,data,'pool5','OutputAs','rows');
    test_x(end+1,:)=actalex;
    test_y(end+1,:)=string2label(char(info.Label));
    count=count+1;
    fprintf('%.2f%% complete...\n',count/8399*100);
end

 train_x=train_x(1:end-1,:);
 train_y=train_y(1:end-1,:);

% Convert data and rescale between 0 and 0.2
train_x = double(train_x) / 255 * 0.2;
test_x  = double(test_x)  / 255 * 0.2;
train_y = double(train_y) * 0.2;
test_y  = double(test_y)  * 0.2;

%% Train SNN
rand('seed', 42);
clear edbn opts;
edbn.sizes = [9216 500 500 10];
opts.numepochs = 2;
opts.alpha = 0.005;
[edbn, opts] = edbnsetup(edbn, opts);

fprintf('Beginning SNN training:\n');
opts.momentum = 0.0; opts.numepochs =  2;
edbn = edbntrain(edbn, train_x, opts);
edbn = edbntoptrain(edbn, train_x, opts, train_y);

opts.momentum = 0.8; opts.numepochs = 60;
edbn = edbntrain(edbn, train_x, opts);
edbn = edbntrain(edbn, train_x, opts);
edbn = edbntoptrain(edbn, train_x, opts, train_y);

fprintf('Hybrid Transfer Learning complete successfully\n');

% Show results
figure;
visualize(edbn.erbm{1}.W');   %  Visualize the RBM weights
er = edbntest (edbn, train_x, train_y);
fprintf('Scored: %2.2f\n', (1-er)*100);
filename = sprintf('good_mnist_%2.2f-%s.mat',(1-er)*100, date());
edbnclean(edbn);
save(filename,'edbn');

% SuperTraining
opts.momentum = 0.8;
opts.numepochs = 80;
edbn = edbntoptrain(edbn, train_x, opts, train_y);

% Show results
figure;
visualize(edbn.erbm{1}.W');   %  Visualize the RBM weights
er = edbntest (edbn, train_x, train_y);
fprintf('Scored: %2.2f\n', (1-er)*100);
filename = sprintf('good_mnist_%2.2f-%s.mat',(1-er)*100, date());
edbnclean(edbn);
save(filename,'edbn');

%% Show the EDBN in action
spike_list = live_edbn(edbn, test_x(1, :), opts);
output_idxs = (spike_list.layers == numel(edbn.sizes));

figure(2); clf;
hist(spike_list.addrs(output_idxs) - 1, 0:edbn.sizes(end));

%% Export to xml to load into JSpikeStack
edbntoxml(edbn, opts, 'mnist_edbn');

%% Auxiliary Functions
function label=string2label(name)
   if strcmp(name, "0")
       label=[1 0 0 0 0 0 0 0 0 0];
   elseif strcmp(name, "1")
       label=[0 1 0 0 0 0 0 0 0 0];
   elseif strcmp(name, "2")
       label=[0 0 1 0 0 0 0 0 0 0];
   elseif strcmp(name, "3")
       label=[0 0 0 1 0 0 0 0 0 0];
   elseif strcmp(name, "4")
       label=[0 0 0 0 1 0 0 0 0 0];
   elseif strcmp(name, "5")
       label=[0 0 0 0 0 1 0 0 0 0];
   elseif strcmp(name, "6")
       label=[0 0 0 0 0 0 1 0 0 0];
   elseif strcmp(name, "7")
       label=[0 0 0 0 0 0 0 1 0 0];
   elseif strcmp(name, "8")
       label=[0 0 0 0 0 0 0 0 1 0];
   elseif strcmp(name, "9")
       label=[0 0 0 0 0 0 0 0 0 1];   
   end
end

function img = digitsread(file)
    % Read in image
    img = imread(file);    
    % Resize to match AlexNet input
    img = imresize(img,[227 227]);
    % Convert grayscale to color (RGB)
    img = repmat(img,[1 1 3]);
end

