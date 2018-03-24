function output = demosaicImage(im, method)
% DEMOSAICIMAGE computes the color image from mosaiced input
%   OUTPUT = DEMOSAICIMAGE(IM, METHOD) computes a demosaiced OUTPUT from
%   the input IM. The choice of the interpolation METHOD can be 
%   'baseline', 'nn', 'linear', 'adagrad'. 

switch lower(method)
    case 'baseline'
        output = demosaicBaseline(im);
    case 'nn'
        output = demosaicNN(im);         % Implement this
    case 'linear'
        output = demosaicLinear(im);     % Implement this
    case 'adagrad'
        output = demosaicAdagrad(im);    % Implement this
end

%--------------------------------------------------------------------------
%                          Baseline demosaicing algorithm. 
%                          The algorithm replaces missing values with the
%                          mean of each color channel.
%--------------------------------------------------------------------------
function mosim = demosaicBaseline(im)
mosim = repmat(im, [1 1 3]); % Create an image by stacking the input
[imageHeight, imageWidth] = size(im);

% Red channel (odd rows and columns);
redValues = im(1:2:imageHeight, 1:2:imageWidth);
meanValue = mean(mean(redValues));
mosim(:,:,1) = meanValue;
mosim(1:2:imageHeight, 1:2:imageWidth,1) = im(1:2:imageHeight, 1:2:imageWidth);

% Blue channel (even rows and colums);
blueValues = im(2:2:imageHeight, 2:2:imageWidth);
meanValue = mean(mean(blueValues));
mosim(:,:,3) = meanValue;
mosim(2:2:imageHeight, 2:2:imageWidth,3) = im(2:2:imageHeight, 2:2:imageWidth);

% Green channel (remaining places)
% We will first create a mask for the green pixels (+1 green, -1 not green)
mask = ones(imageHeight, imageWidth);
mask(1:2:imageHeight, 1:2:imageWidth) = -1;
mask(2:2:imageHeight, 2:2:imageWidth) = -1;
greenValues = mosim(mask > 0);
meanValue = mean(greenValues);
% For the green pixels we copy the value
greenChannel = im;
greenChannel(mask < 0) = meanValue;
mosim(:,:,2) = greenChannel;


%--------------------------------------------------------------------------
%                           Nearest neighbour algorithm
%--------------------------------------------------------------------------
function Channels = demosaicNN(im)
%
% Implement this 
%
mosim = repmat(im, [1 1 3]); % Create an image by stacking the input
[imageHeight, imageWidth] = size(im);

% Red channel (odd rows and columns);
mosim(:,:,1) = 0;
mosim(1:2:imageHeight, 1:2:imageWidth,1) = im(1:2:imageHeight, 1:2:imageWidth);

% Blue channel (even rows and colums);
mosim(:,:,3) = 0;
mosim(2:2:imageHeight, 2:2:imageWidth,3) = im(2:2:imageHeight, 2:2:imageWidth);

% Green channel (remaining places)
% We will first create a mask for the green pixels (+1 green, -1 not green)
mask = ones(imageHeight, imageWidth);
mask(1:2:imageHeight, 1:2:imageWidth) = -1;
mask(2:2:imageHeight, 2:2:imageWidth) = -1;
% For the green pixels we copy the value
greenChannel = im;
greenChannel(mask < 0) = 0;
mosim(:,:,2) = greenChannel;
m = imageHeight;
n = imageWidth;
basicZeros = zeros(imageHeight, imageWidth);
Rc= basicZeros;
Gc= basicZeros;
Bc= basicZeros;

Rc(2:m-1, 2:n-1) = conv2(mosim(:,:,1),[0 0 0; 0 1 1; 0 1 1],'valid');
Gc(2:m-1, 2:n-1) = conv2(mosim(:,:,2), [0 0 0; 0 1 1; 0 0 0],'valid');
Bc(2:m-1, 2:n-1) = conv2(mosim(:,:,3),[0 0 0; 0 1 1; 0 1 1],'valid');
% bottom odd, right odd
if ((mod(m,2) == 1)&&(mod(n,2) == 1))
    RedLT = conv2(mosim(:,:,1),[0 0 0; 0 1 1; 0 1 0],'same');
    RedR = conv2(mosim(:,:,1),[0 0 0; 0 1 0; 0 1 0],'same');
    RedB = conv2(mosim(:,:,1),[0 0 0; 0 1 1; 0 0 0],'same');
    GreenLT = conv2(mosim(:,:,2),[0 0 0; 0 1 1; 0 1 0],'same');
    GreenR = conv2(mosim(:,:,2),[0 0 0; 0 1 1; 0 0 0],'same');
    GreenB = conv2(mosim(:,:,2),[0 0 0; 0 1 1; 0 0 0],'same');
    BlueLT = conv2(mosim(:,:,3),[0 1 1; 1 0 0; 1 0 0],'same');
    BlueR = conv2(mosim(:,:,3),[0 0 0; 0 0 1; 0 0 1],'same');
    BlueB = conv2(mosim(:,:,3),[0 0 0; 0 0 0; 0 1 1],'same');
    Channels = zeros(imageHeight, imageWidth,3);
    Channels(1,:,1) = RedLT(1,:);
    Channels(:,1,1) = RedLT(:,1);
    Channels(end,:,1) = RedB(end,:);
    Channels(:,end,1) = RedR(:,end);
    Channels(:,:,1) = Channels(:,:,1) + Rc;
    
    Channels(1,:,2) = GreenLT(1,:);
    Channels(:,1,2) = GreenLT(:,1);
    Channels(end,:,2) = GreenB(end,:);
    Channels(:,end,2) = GreenR(:,end);
    Channels(:,:,2) = Channels(:,:,2) + Gc;
    
    Channels(1,:,3) = BlueLT(1,:);
    Channels(:,1,3) = BlueLT(:,1);
    Channels(end,:,3) = BlueB(end,:);
    Channels(:,end,3) = BlueR(:,end);
    Channels(:,:,3) = Channels(:,:,3) + Bc;
end
% bottom odd, right even
if ((mod(m,2) == 1)&&(mod(n,2) == 0))
    RedLT = conv2(mosim(:,:,1),[0 0 0; 0 1 1; 0 1 0],'same');
    RedR = conv2(mosim(:,:,1),[0 0 0; 0 0 1; 0 0 1],'same');
    RedB = conv2(mosim(:,:,1),[0 0 0; 0 1 1; 0 0 0],'same');
    GreenLT = conv2(mosim(:,:,2),[0 0 0; 0 1 1; 0 1 0],'same');
    GreenR = conv2(mosim(:,:,2),[0 0 0; 0 1 1; 0 0 0],'same');
    GreenB = conv2(mosim(:,:,2),[0 0 0; 0 1 1; 0 0 0],'same');
    BlueLT = conv2(mosim(:,:,3),[0 1 1; 1 0 0; 1 0 0],'same');
    BlueR = conv2(mosim(:,:,3),[0 0 0; 0 1 0; 0 1 0],'same');
    BlueB = conv2(mosim(:,:,3),[0 0 0; 0 0 0; 0 1 1],'same');
    Channels = zeros(imageHeight, imageWidth,3);
    Channels(1,:,1) = RedLT(1,:);
    Channels(:,1,1) = RedLT(:,1);
    Channels(end,:,1) = RedB(end,:);
    Channels(:,end,1) = RedR(:,end);
    Channels(:,:,1) = Channels(:,:,1) + Rc;
    
    Channels(1,:,2) = GreenLT(1,:);
    Channels(:,1,2) = GreenLT(:,1);
    Channels(end,:,2) = GreenB(end,:);
    Channels(:,end,2) = GreenR(:,end);
    Channels(:,:,2) = Channels(:,:,2) + Gc;
    
    Channels(1,:,3) = BlueLT(1,:);
    Channels(:,1,3) = BlueLT(:,1);
    Channels(end,:,3) = BlueB(end,:);
    Channels(:,end,3) = BlueR(:,end);
    Channels(:,:,3) = Channels(:,:,3) + Bc;
end
% bottom even, right odd
if ((mod(m,2) == 0)&&(mod(n,2) == 1))
    RedLT = conv2(mosim(:,:,1),[0 0 0; 0 1 1; 0 1 0],'same');
    RedR = conv2(mosim(:,:,1),[0 0 0; 0 1 0; 0 1 0],'same');
    RedB = conv2(mosim(:,:,1),[0 0 0; 0 0 0; 0 1 1],'same');
    GreenLT = conv2(mosim(:,:,2),[0 0 0; 0 1 1; 0 1 0],'same');
    GreenR = conv2(mosim(:,:,2),[0 0 0; 0 1 1; 0 0 0],'same');
    GreenB = conv2(mosim(:,:,2),[0 0 0; 0 1 1; 0 0 0],'same');
    BlueLT = conv2(mosim(:,:,3),[0 1 1; 1 0 0; 1 0 0],'same');
    BlueR = conv2(mosim(:,:,3),[0 0 0; 0 0 1; 0 0 1],'same');
    BlueB = conv2(mosim(:,:,3),[0 0 0; 0 1 1; 0 0 0],'same');
    Channels = zeros(imageHeight, imageWidth,3);
    Channels(1,:,1) = RedLT(1,:);
    Channels(:,1,1) = RedLT(:,1);
    Channels(end,:,1) = RedB(end,:);
    Channels(:,end,1) = RedR(:,end);
    Channels(:,:,1) = Channels(:,:,1) + Rc;
    
    Channels(1,:,2) = GreenLT(1,:);
    Channels(:,1,2) = GreenLT(:,1);
    Channels(end,:,2) = GreenB(end,:);
    Channels(:,end,2) = GreenR(:,end);
    Channels(:,:,2) = Channels(:,:,2) + Gc;
    
    Channels(1,:,3) = BlueLT(1,:);
    Channels(:,1,3) = BlueLT(:,1);
    Channels(end,:,3) = BlueB(end,:);
    Channels(:,end,3) = BlueR(:,end);
    Channels(:,:,3) = Channels(:,:,3) + Bc;
end
% bottom even, right even
if ((mod(m,2) == 0)&&(mod(n,2) == 0))
    RedLT = conv2(mosim(:,:,1),[0 0 0; 0 1 1; 0 1 0],'same');
    RedR = conv2(mosim(:,:,1),[0 0 0; 0 0 1; 0 0 1],'same');
    RedB = conv2(mosim(:,:,1),[0 0 0; 0 0 0; 0 1 1],'same');
    GreenLT = conv2(mosim(:,:,2),[0 0 0; 0 1 1; 0 1 0],'same');
    GreenR = conv2(mosim(:,:,2),[0 0 0; 0 1 1; 0 0 0],'same');
    GreenB = conv2(mosim(:,:,2),[0 0 0; 0 1 1; 0 0 0],'same');
    BlueLT = conv2(mosim(:,:,3),[0 1 1; 1 0 0; 1 0 0],'same');
    BlueR = conv2(mosim(:,:,3),[0 0 0; 0 1 0; 0 1 0],'same');
    BlueB = conv2(mosim(:,:,3),[0 0 0; 0 1 1; 0 0 0],'same');
    Channels = zeros(imageHeight, imageWidth,3);
    Channels(1,:,1) = RedLT(1,:);
    Channels(:,1,1) = RedLT(:,1);
    Channels(end,:,1) = RedB(end,:);
    Channels(:,end,1) = RedR(:,end);
    Channels(:,:,1) = Channels(:,:,1) + Rc;
    
    Channels(1,:,2) = GreenLT(1,:);
    Channels(:,1,2) = GreenLT(:,1);
    Channels(end,:,2) = GreenB(end,:);
    Channels(:,end,2) = GreenR(:,end);
    Channels(:,:,2) = Channels(:,:,2) + Gc;
    
    Channels(1,:,3) = BlueLT(1,:);
    Channels(:,1,3) = BlueLT(:,1);
    Channels(end,:,3) = BlueB(end,:);
    Channels(:,end,3) = BlueR(:,end);
    Channels(:,:,3) = Channels(:,:,3) + Bc;
end

Channels(1,1,2) = mosim(1,2,2);
Channels(1,1,3) = mosim(2,2,3);
if mod(m,2) == 1 && mod(n,2) == 1
    Channels(1,n,2) = mosim(1,n-1,2);
    Channels(1,n,3) = mosim(2,n-1,3);
    Channels(m,1,2) = mosim(m-1,1,2);
    Channels(m,1,3) = mosim(m-1,2,3);
    Channels(m,n,2) = mosim(m-1,n,2);
    Channels(m,n,3) = mosim(m-1,n-1,3);
end
if mod(m,2) == 0 && mod(n,2) == 1
    Channels(1,n,2) = mosim(1,n-1,2);
    Channels(1,n,3) = mosim(2,n-1,3);
    Channels(m,1,1) = mosim(m-1,1,1);
    Channels(m,1,3) = mosim(m,2,3);
    Channels(m,n,1) = mosim(m-1,n,1);
    Channels(m,n,3) = mosim(m,n-1,3);
end
if mod(m,2) == 1 && mod(n,2) == 0
    Channels(1,n,1) = mosim(1,n-1,1);
    Channels(1,n,3) = mosim(2,n,3);
    Channels(m,1,2) = mosim(m-1,1,2);
    Channels(m,1,3) = mosim(m-1,2,3);
    Channels(m,n,1) = mosim(m-1,n-1,1);
    Channels(m,n,2) = mosim(m-1,n,2);
end
if mod(m,2) == 0 && mod(n,2) == 0
    Channels(1,n,1) = mosim(1,n-1,1);
    Channels(1,n,3) = mosim(2,n,3);
    Channels(m,1,1) = mosim(m-1,1,1);
    Channels(m,1,3) = mosim(m,2,3);
    Channels(m,n,1) = mosim(m-1,n-1,1);
    Channels(m,n,2) = mosim(m-1,n,2);
end

%--------------------------------------------------------------------------
%                           Linear interpolation
%--------------------------------------------------------------------------
function Channels = demosaicLinear(im)
%
% Implement this 
%
mosim = repmat(im, [1 1 3]); % Create an image by stacking the input
[imageHeight, imageWidth] = size(im);

% Red channel (odd rows and columns);
mosim(:,:,1) = 0;
mosim(1:2:imageHeight, 1:2:imageWidth,1) = im(1:2:imageHeight, 1:2:imageWidth);

% Blue channel (even rows and colums);
mosim(:,:,3) = 0;
mosim(2:2:imageHeight, 2:2:imageWidth,3) = im(2:2:imageHeight, 2:2:imageWidth);

% Green channel (remaining places)
% We will first create a mask for the green pixels (+1 green, -1 not green)
mask = ones(imageHeight, imageWidth);
mask(1:2:imageHeight, 1:2:imageWidth) = -1;
mask(2:2:imageHeight, 2:2:imageWidth) = -1;
% For the green pixels we copy the value
greenChannel = im;
greenChannel(mask < 0) = 0;
mosim(:,:,2) = greenChannel;
m = imageHeight;
n = imageWidth;
basicZeros = zeros(imageHeight, imageWidth);
Rc= basicZeros;
Gc= basicZeros;
Bc= basicZeros;

Rc(2:m-1, 2:n-1) = conv2(mosim(:,:,1),[1/4 1/2 1/4; 1/2 1 1/2; 1/4 1/2 1/4],'valid');
Gc(2:m-1, 2:n-1) = conv2(mosim(:,:,2),[0 1/4 0; 1/4 1 1/4; 0 1/4 0],'valid');
Bc(2:m-1, 2:n-1) = conv2(mosim(:,:,3), [1/4 1/2 1/4; 1/2 1 1/2; 1/4 1/2 1/4],'valid');
RedLT = conv2(mosim(:,:,1),[0 1/2 0; 1/2 1 1/2; 0 1/2 0],'same');
GreenEdge = conv2(mosim(:,:,2),[0 1/3 0; 1/3 1 1/3; 0 1/3 0],'same');
BlueLT = conv2(mosim(:,:,3),[1/2 1 1/2; 1 0 0; 1/2 0 0],'same');
% bottom odd, right odd
if ((mod(m,2) == 1)&&(mod(n,2) == 1))
    RedR = conv2(mosim(:,:,1),[0 1/2 0; 0 1 0; 0 1/2 0],'same');
    RedB = conv2(mosim(:,:,1),[0 0 0; 1/2 1 1/2; 0 0 0],'same');
    BlueR = conv2(mosim(:,:,3),[0 0 1/2; 0 0 1; 0 0 1/2],'same');
    BlueB = conv2(mosim(:,:,3),[0 0 0; 0 0 0; 1/2 1 1/2],'same');
    Channels = zeros(imageHeight, imageWidth,3);
    Channels(1,:,1) = RedLT(1,:);
    Channels(:,1,1) = RedLT(:,1);
    Channels(end,:,1) = RedB(end,:);
    Channels(:,end,1) = RedR(:,end);
    Channels(:,:,1) = Channels(:,:,1) + Rc;
    
    Channels(1,:,2) = GreenEdge(1,:);
    Channels(:,1,2) = GreenEdge(:,1);
    Channels(end,:,2) = GreenEdge(end,:);
    Channels(:,end,2) = GreenEdge(:,end);
    Channels(:,:,2) = Channels(:,:,2) + Gc;
    
    Channels(1,:,3) = BlueLT(1,:);
    Channels(:,1,3) = BlueLT(:,1);
    Channels(end,:,3) = BlueB(end,:);
    Channels(:,end,3) = BlueR(:,end);
    Channels(:,:,3) = Channels(:,:,3) + Bc;
end
% bottom odd, right even
if ((mod(m,2) == 1)&&(mod(n,2) == 0))
    RedR = conv2(mosim(:,:,1),[0 0 1/2; 0 0 1; 0 0 1/2],'same');
    RedB = conv2(mosim(:,:,1),[0 0 0; 1/2 1 1/2; 0 0 0],'same');
    BlueR = conv2(mosim(:,:,3),[0 1/2 0; 0 1 0; 0 1/2 0],'same');
    BlueB = conv2(mosim(:,:,3),[0 0 0; 0 0 0; 1/2 1 1/2],'same');
    Channels = zeros(imageHeight, imageWidth,3);
    Channels(1,:,1) = RedLT(1,:);
    Channels(:,1,1) = RedLT(:,1);
    Channels(end,:,1) = RedB(end,:);
    Channels(:,end,1) = RedR(:,end);
    Channels(:,:,1) = Channels(:,:,1) + Rc;
    
    Channels(1,:,2) = GreenEdge(1,:);
    Channels(:,1,2) = GreenEdge(:,1);
    Channels(end,:,2) = GreenEdge(end,:);
    Channels(:,end,2) = GreenEdge(:,end);
    Channels(:,:,2) = Channels(:,:,2) + Gc;
    
    Channels(1,:,3) = BlueLT(1,:);
    Channels(:,1,3) = BlueLT(:,1);
    Channels(end,:,3) = BlueB(end,:);
    Channels(:,end,3) = BlueR(:,end);
    Channels(:,:,3) = Channels(:,:,3) + Bc;
end
% bottom even, right odd
if ((mod(m,2) == 0)&&(mod(n,2) == 1))
    RedR = conv2(mosim(:,:,1),[0 1/2 0; 0 1 0; 0 1/2 0],'same');
    RedB = conv2(mosim(:,:,1),[0 0 0; 0 0 0; 1/2 1 1/2],'same');
    BlueR = conv2(mosim(:,:,3),[0 0 1/2; 0 0 1; 0 0 1/2],'same');
    BlueB = conv2(mosim(:,:,3),[0 0 0; 1/2 1 1/2; 0 0 0],'same');
    Channels = zeros(imageHeight, imageWidth,3);
    Channels(1,:,1) = RedLT(1,:);
    Channels(:,1,1) = RedLT(:,1);
    Channels(end,:,1) = RedB(end,:);
    Channels(:,end,1) = RedR(:,end);
    Channels(:,:,1) = Channels(:,:,1) + Rc;
    
    Channels(1,:,2) = GreenEdge(1,:);
    Channels(:,1,2) = GreenEdge(:,1);
    Channels(end,:,2) = GreenEdge(end,:);
    Channels(:,end,2) = GreenEdge(:,end);
    Channels(:,:,2) = Channels(:,:,2) + Gc;
    
    Channels(1,:,3) = BlueLT(1,:);
    Channels(:,1,3) = BlueLT(:,1);
    Channels(end,:,3) = BlueB(end,:);
    Channels(:,end,3) = BlueR(:,end);
    Channels(:,:,3) = Channels(:,:,3) + Bc;
end
% bottom even, right even
if ((mod(m,2) == 0)&&(mod(n,2) == 0))
    RedR = conv2(mosim(:,:,1),[0 0 1/2; 0 0 1/2; 0 0 1/2],'same');
    RedB = conv2(mosim(:,:,1),[0 0 0; 0 0 0; 1/2 1 1/2],'same');
    BlueR = conv2(mosim(:,:,3),[0 1/2 0; 0 1 0; 0 1/2 0],'same');
    BlueB = conv2(mosim(:,:,3),[0 0 0; 1/2 1 1/2; 0 0 0],'same');
    Channels = zeros(imageHeight, imageWidth,3);
    Channels(1,:,1) = RedLT(1,:);
    Channels(:,1,1) = RedLT(:,1);
    Channels(end,:,1) = RedB(end,:);
    Channels(:,end,1) = RedR(:,end);
    Channels(:,:,1) = Channels(:,:,1) + Rc;
    
    Channels(1,:,2) = GreenEdge(1,:);
    Channels(:,1,2) = GreenEdge(:,1);
    Channels(end,:,2) = GreenEdge(end,:);
    Channels(:,end,2) = GreenEdge(:,end);
    Channels(:,:,2) = Channels(:,:,2) + Gc;
    
    Channels(1,:,3) = BlueLT(1,:);
    Channels(:,1,3) = BlueLT(:,1);
    Channels(end,:,3) = BlueB(end,:);
    Channels(:,end,3) = BlueR(:,end);
    Channels(:,:,3) = Channels(:,:,3) + Bc;
end

% Calculate Corners
Channels(1,1,2) = (mosim(1,2,2)+mosim(2,1,2))/2;
Channels(1,1,3) = mosim(2,2,3);
if mod(m,2) == 1 && mod(n,2) == 1
    Channels(1,n,2) = (mosim(1,n-1,2)+mosim(2,n,2))/2;
    Channels(1,n,3) = mosim(2,n-1,3);
    Channels(m,1,2) = (mosim(m-1,1,2)+mosim(m,2,2))/2;
    Channels(m,1,3) = mosim(m-1,2,3);
    Channels(m,n,2) = (mosim(m-1,n,2)+mosim(m,n-1,2))/2;
    Channels(m,n,3) = mosim(m-1,n-1,3);
end
if mod(m,2) == 0 && mod(n,2) == 1
    Channels(1,n,2) = (mosim(1,n-1,2)+mosim(2,n,2))/2;
    Channels(1,n,3) = mosim(2,n-1,3);
    Channels(m,1,1) = mosim(m-1,1,1);
    Channels(m,1,3) = mosim(m,2,3);
    Channels(m,n,1) = mosim(m-1,n,1);
    Channels(m,n,3) = mosim(m,n-1,3);
end
if mod(m,2) == 1 && mod(n,2) == 0
    Channels(1,n,1) = mosim(1,n-1,1);
    Channels(1,n,3) = mosim(2,n,3);
    Channels(m,1,2) = (mosim(m-1,1,2)+mosim(m,2,2))/2;
    Channels(m,1,3) = mosim(m-1,2,3);
    Channels(m,n,1) = mosim(m-1,n-1,1);
    Channels(m,n,2) = mosim(m-1,n,2);
end
if mod(m,2) == 0 && mod(n,2) == 0
    Channels(1,n,2) = (mosim(1,n-1,2)+mosim(2,n,2))/2;
    Channels(1,n,3) = mosim(2,n-1,3);
    Channels(m,1,2) = (mosim(m-1,1,2)+mosim(m,2,2))/2;
    Channels(m,1,3) = mosim(m-1,2,3);
    Channels(m,n,1) = mosim(m-1,n-1,1);
    Channels(m,n,2) = (mosim(m-1,n,2)+mosim(m,n-1,2))/2;
end

%--------------------------------------------------------------------------
%                           Adaptive gradient
%--------------------------------------------------------------------------
function Channels = demosaicAdagrad(im)
%
% Implement this 
mosim = repmat(im, [1 1 3]); % Create an image by stacking the input
[imageHeight, imageWidth] = size(im);

% Red channel (odd rows and columns);
mosim(:,:,1) = 0;
mosim(1:2:imageHeight, 1:2:imageWidth,1) = im(1:2:imageHeight, 1:2:imageWidth);

% Blue channel (even rows and colums);
mosim(:,:,3) = 0;
mosim(2:2:imageHeight, 2:2:imageWidth,3) = im(2:2:imageHeight, 2:2:imageWidth);

% Green channel (remaining places)
% We will first create a mask for the green pixels (+1 green, -1 not green)
mask = ones(imageHeight, imageWidth);
mask(1:2:imageHeight, 1:2:imageWidth) = -1;
mask(2:2:imageHeight, 2:2:imageWidth) = -1;
% For the green pixels we copy the value
greenChannel = im;
greenChannel(mask < 0) = 0;
mosim(:,:,2) = greenChannel;
m = imageHeight;
n = imageWidth;
basicZeros = zeros(imageHeight-2, imageWidth-2);
Rc = basicZeros;
Gc = basicZeros;
Bc = basicZeros;
RcV = conv2(mosim(:,:,1),[1/2 1/2 0; 1/2 1 1/2; 0 1/2 1/2],'valid');
RcCompV = conv2(mosim(:,:,1),[1/2 1/2 0; 1/2 1 1/2; 0 1/2 -1/2],'valid');
RcH = conv2(mosim(:,:,1),[0 1/2 1/2; 1/2 1 1/2; 1/2 1/2 0],'valid');
RcCompH = conv2(mosim(:,:,1),[0 1/2 -1/2; 1/2 1 1/2; 1/2 1/2 0],'valid');
RcMask = abs(RcCompV) - abs(RcCompH);
Rc(RcMask<=0) = RcV(RcMask<=0);
Rc(RcMask>0) = RcH(RcMask>0);

GcV = conv2(mosim(:,:,2),[0 1/2 0; 0 1 0; 0 1/2 0],'valid');
GcCompV = conv2(mosim(:,:,2),[0 -1/2 0; 0 1 0; 0 1/2 0],'valid');
GcH = conv2(mosim(:,:,2),[0 0 0; 1/2 1 1/2; 0 0 0],'valid');
GcCompH = conv2(mosim(:,:,2),[0 0 0; -1/2 1 1/2; 0 0 0],'valid');
GcMask = abs(GcCompV) - abs(GcCompH);
Gc(GcMask<=0) = GcV(GcMask<=0);
Gc(GcMask>0) = GcH(GcMask>0);

BcV = conv2(mosim(:,:,3),[1/2 1/2 0; 1/2 1 1/2; 0 1/2 1/2],'valid');
BcCompV = conv2(mosim(:,:,3),[1/2 1/2 0; 1/2 1 1/2; 0 1/2 -1/2],'valid');
BcH = conv2(mosim(:,:,3),[0 1/2 1/2; 1/2 1 1/2; 1/2 1/2 0],'valid');
BcCompH = conv2(mosim(:,:,3),[0 1/2 -1/2; 1/2 1 1/2; 1/2 1/2 0],'valid');
BcMask = abs(BcCompV) - abs(BcCompH);
Bc(BcMask<=0) = BcV(BcMask<=0);
Bc(BcMask>0) = BcH(BcMask>0);

RedLT = conv2(mosim(:,:,1),[0 1/2 0; 1/2 1 1/2; 0 1/2 0],'same');
GreenEdge = conv2(mosim(:,:,2),[0 1/3 0; 1/3 1 1/3; 0 1/3 0],'same');
BlueLT = conv2(mosim(:,:,3),[1/2 1 1/2; 1 0 0; 1/2 0 0],'same');
% bottom odd, right odd
if ((mod(m,2) == 1)&&(mod(n,2) == 1))
    RedR = conv2(mosim(:,:,1),[0 1/2 0; 0 1 0; 0 1/2 0],'same');
    RedB = conv2(mosim(:,:,1),[0 0 0; 1/2 1 1/2; 0 0 0],'same');
    BlueR = conv2(mosim(:,:,3),[0 0 1/2; 0 0 1; 0 0 1/2],'same');
    BlueB = conv2(mosim(:,:,3),[0 0 0; 0 0 0; 1/2 1 1/2],'same');
    Channels = zeros(imageHeight, imageWidth,3);
    Channels(1,:,1) = RedLT(1,:);
    Channels(:,1,1) = RedLT(:,1);
    Channels(end,:,1) = RedB(end,:);
    Channels(:,end,1) = RedR(:,end);
    Channels(2:m-1,2:n-1,1) = Rc;
    
    Channels(1,:,2) = GreenEdge(1,:);
    Channels(:,1,2) = GreenEdge(:,1);
    Channels(end,:,2) = GreenEdge(end,:);
    Channels(:,end,2) = GreenEdge(:,end);
    Channels(2:m-1,2:n-1,2) = Gc;
    
    Channels(1,:,3) = BlueLT(1,:);
    Channels(:,1,3) = BlueLT(:,1);
    Channels(end,:,3) = BlueB(end,:);
    Channels(:,end,3) = BlueR(:,end);
    Channels(2:m-1,2:n-1,3) = Bc;
end
% bottom odd, right even
if ((mod(m,2) == 1)&&(mod(n,2) == 0))
    RedR = conv2(mosim(:,:,1),[0 0 1/2; 0 0 1; 0 0 1/2],'same');
    RedB = conv2(mosim(:,:,1),[0 0 0; 1/2 1 1/2; 0 0 0],'same');
    BlueR = conv2(mosim(:,:,3),[0 1/2 0; 0 1 0; 0 1/2 0],'same');
    BlueB = conv2(mosim(:,:,3),[0 0 0; 0 0 0; 1/2 1 1/2],'same');
    Channels = zeros(imageHeight, imageWidth,3);
    Channels(1,:,1) = RedLT(1,:);
    Channels(:,1,1) = RedLT(:,1);
    Channels(end,:,1) = RedB(end,:);
    Channels(:,end,1) = RedR(:,end);
    Channels(2:m-1,2:n-1,1) = Rc;
    
    Channels(1,:,2) = GreenEdge(1,:);
    Channels(:,1,2) = GreenEdge(:,1);
    Channels(end,:,2) = GreenEdge(end,:);
    Channels(:,end,2) = GreenEdge(:,end);
    Channels(2:m-1,2:n-1,2) = Gc;
    
    Channels(1,:,3) = BlueLT(1,:);
    Channels(:,1,3) = BlueLT(:,1);
    Channels(end,:,3) = BlueB(end,:);
    Channels(:,end,3) = BlueR(:,end);
    Channels(2:m-1,2:n-1,3) = Bc;
end
% bottom even, right odd
if ((mod(m,2) == 0)&&(mod(n,2) == 1))
    RedR = conv2(mosim(:,:,1),[0 1/2 0; 0 1 0; 0 1/2 0],'same');
    RedB = conv2(mosim(:,:,1),[0 0 0; 0 0 0; 1/2 1 1/2],'same');
    BlueR = conv2(mosim(:,:,3),[0 0 1/2; 0 0 1; 0 0 1/2],'same');
    BlueB = conv2(mosim(:,:,3),[0 0 0; 1/2 1 1/2; 0 0 0],'same');
    Channels = zeros(imageHeight, imageWidth,3);
    Channels(1,:,1) = RedLT(1,:);
    Channels(:,1,1) = RedLT(:,1);
    Channels(end,:,1) = RedB(end,:);
    Channels(:,end,1) = RedR(:,end);
    Channels(2:m-1,2:n-1,1) = Rc;
    
    Channels(1,:,2) = GreenEdge(1,:);
    Channels(:,1,2) = GreenEdge(:,1);
    Channels(end,:,2) = GreenEdge(end,:);
    Channels(:,end,2) = GreenEdge(:,end);
    Channels(2:m-1,2:n-1,2) = Gc;
    
    Channels(1,:,3) = BlueLT(1,:);
    Channels(:,1,3) = BlueLT(:,1);
    Channels(end,:,3) = BlueB(end,:);
    Channels(:,end,3) = BlueR(:,end);
    Channels(2:m-1,2:n-1,3) = Bc;
end
% bottom even, right even
if ((mod(m,2) == 0)&&(mod(n,2) == 0))
    RedR = conv2(mosim(:,:,1),[0 0 1/2; 0 0 1/2; 0 0 1/2],'same');
    RedB = conv2(mosim(:,:,1),[0 0 0; 0 0 0; 1/2 1 1/2],'same');
    BlueR = conv2(mosim(:,:,3),[0 1/2 0; 0 1 0; 0 1/2 0],'same');
    BlueB = conv2(mosim(:,:,3),[0 0 0; 1/2 1 1/2; 0 0 0],'same');
    Channels = zeros(imageHeight, imageWidth,3);
    Channels(1,:,1) = RedLT(1,:);
    Channels(:,1,1) = RedLT(:,1);
    Channels(end,:,1) = RedB(end,:);
    Channels(:,end,1) = RedR(:,end);
    Channels(2:m-1,2:n-1,1) = Rc;
    
    Channels(1,:,2) = GreenEdge(1,:);
    Channels(:,1,2) = GreenEdge(:,1);
    Channels(end,:,2) = GreenEdge(end,:);
    Channels(:,end,2) = GreenEdge(:,end);
    Channels(2:m-1,2:n-1,2) = Gc;
    
    Channels(1,:,3) = BlueLT(1,:);
    Channels(:,1,3) = BlueLT(:,1);
    Channels(end,:,3) = BlueB(end,:);
    Channels(:,end,3) = BlueR(:,end);
    Channels(2:m-1,2:n-1,3) = Bc;
end
% Calculate Corners
Channels(1,1,2) = (mosim(1,2,2)+mosim(2,1,2))/2;
Channels(1,1,3) = mosim(2,2,3);
if mod(m,2) == 1 && mod(n,2) == 1
    Channels(1,n,2) = (mosim(1,n-1,2)+mosim(2,n,2))/2;
    Channels(1,n,3) = mosim(2,n-1,3);
    Channels(m,1,2) = (mosim(m-1,1,2)+mosim(m,2,2))/2;
    Channels(m,1,3) = mosim(m-1,2,3);
    Channels(m,n,2) = (mosim(m-1,n,2)+mosim(m,n-1,2))/2;
    Channels(m,n,3) = mosim(m-1,n-1,3);
end
if mod(m,2) == 0 && mod(n,2) == 1
    Channels(1,n,2) = (mosim(1,n-1,2)+mosim(2,n,2))/2;
    Channels(1,n,3) = mosim(2,n-1,3);
    Channels(m,1,1) = mosim(m-1,1,1);
    Channels(m,1,3) = mosim(m,2,3);
    Channels(m,n,1) = mosim(m-1,n,1);
    Channels(m,n,3) = mosim(m,n-1,3);
end
if mod(m,2) == 1 && mod(n,2) == 0
    Channels(1,n,1) = mosim(1,n-1,1);
    Channels(1,n,3) = mosim(2,n,3);
    Channels(m,1,2) = (mosim(m-1,1,2)+mosim(m,2,2))/2;
    Channels(m,1,3) = mosim(m-1,2,3);
    Channels(m,n,1) = mosim(m-1,n-1,1);
    Channels(m,n,2) = mosim(m-1,n,2);
end
if mod(m,2) == 0 && mod(n,2) == 0
    Channels(1,n,2) = (mosim(1,n-1,2)+mosim(2,n,2))/2;
    Channels(1,n,3) = mosim(2,n-1,3);
    Channels(m,1,2) = (mosim(m-1,1,2)+mosim(m,2,2))/2;
    Channels(m,1,3) = mosim(m-1,2,3);
    Channels(m,n,1) = mosim(m-1,n-1,1);
    Channels(m,n,2) = (mosim(m-1,n,2)+mosim(m,n-1,2))/2;
end