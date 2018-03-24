function [imShift, predShift] = alignChannels(im, maxShift)
% ALIGNCHANNELS align channels in an image.
%   [IMSHIFT, PREDSHIFT] = ALIGNCHANNELS(IM, MAXSHIFT) aligns the channels in an
%   NxMx3 image IM. The first channel is fixed and the remaining channels
%   are aligned to it within the maximum displacement range of MAXSHIFT (in
%   both directions). The code returns the aligned image IMSHIFT after
%   performing this alignment. The optimal shifts are returned as in
%   PREDSHIFT a 2x2 array. PREDSHIFT(1,:) is the shifts  in I (the first) 
%   and J (the second) dimension of the second channel, and PREDSHIFT(2,:)
%   are the same for the third channel.


% Sanity check
assert(size(im,3) == 3);
assert(all(maxShift > 0));

%Find edges
% imedge1 = edge(im(:,:,1),'canny');
% imedge2 = edge(im(:,:,2),'canny');
% imedge3 = edge(im(:,:,3),'canny');

% Dummy implementation (replace this with your own)
predShift = zeros(2, 2);
a = maxShift(1);
b = maxShift(2);
% coarse to fine

max = 0;
for i = -a:a
    for j = -b:b
        if sum(sum(circshift(im(:,:,2),[i j]).*im(:,:,1)))>max
            predShift(1,:)=[i j];
            max = sum(sum(circshift(im(:,:,2),[i j]).*im(:,:,1)));
        end
    end
end

max = 0;
for i = -a:a
    for j = -b:b
        if sum(sum(circshift(im(:,:,3),[i j]).*im(:,:,1)))>max
            predShift(2,:)=[i j];
            max = sum(sum(circshift(im(:,:,3),[i j]).*im(:,:,1)));
        end
    end
end



imShift = im;
imShift(:,:,2) = circshift(imShift(:,:,2),predShift(1,:));
imShift(:,:,3) = circshift(imShift(:,:,3),predShift(2,:));
