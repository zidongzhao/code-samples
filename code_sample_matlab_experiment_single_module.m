%$ with contribution by Mike Morais
%$%$ pull store matlab objects from the day prior to make randomization consistent across the two days-------------------------------------
prev_day = dir(sprintf('%s_*_func_task-familiarization_run-03.mat',subid));
if size(prev_day,1) == 1
  prev_day = prev_day.name;
  load(prev_day);
  %$%$ make sure to always load imdata before des so the new des file overwrites the old one stored in imdata
  load('imdata.mat');
  %% import design matrix
  des = importdata('stimulidesign.xlsx'); des = des.all;
  des_trial = des(1:6,:);
  des = des(7:end,:);
  tlog = [];
else
  stcbpe_taskcache;
  tlog = [];
end

set(0,'DefaultAxesFontUnits','pixels');


% First clear all variables and close open windows
clearvars -except imdata* imfnm imdir P des* extra* out outref tlog stims demoversion fmri_autorun interrupted once_interrupted;
close all;
home
Pcopy = P;

%% Boot up Psychtoolbox

% useful to include when coding up experiment, skips all synchronisation tests.
PsychDebugWindowConfiguration(0,1)
Screen('Preference', 'SkipSyncTests', 1);

% if there is a second monitor psychtoolbox will use this display
defaults.scrID      = max(Screen('Screens'));
% open the psychtoolbox window on the second monitor
[w, winRect]        = PsychImaging('OpenWindow',defaults.scrID,[128 128 128]);

%$%$ This might need to be modified to suit the audio system on the scanner computer
% open audio channels
%       Note: see http://www.lrdc.pitt.edu/maplelab/matlab_audio.html for
%       tips ging and setting up!
InitializePsychSound(1);
        % If you're on a Windows machine, it REALLY doesn't like opening
        % multiple ports. Linux machines don't care.
pahandle_a = PsychPortAudio('Open', [], 1, [], 44100, 1, [], 0.015);
pahandle_b = PsychPortAudio('Open', [], 1, [], 44100, 1, [], 0.015);
pahandle_c = PsychPortAudio('Open', [], 1, [], 44100, 1, [], 0.015);

%$%$ keyboard response setup modified by Z, with EAM's code, for scanner
% %% Set valid keyboard responses
% activeKeys = [82 81];
% keylookup = uint16(zeros([1 256]));
% keylookup([82 81]) = [0 1]; % Up and down arrows
% keylookup(30:36) = 1:7; % Numbers 1-7
%
% key.dev_id      =   GetKeyboardIndices; % device id for checking responses.  -1 for all devices (but might be slow and miss some responses)
% key.trigger     =   34;
% key.topchoice   =   82; % Up Arrow
% key.botchoice   =   81; % Down Arrow
% key.nums        =   30:33;
key.nums     =   [KbName('1!') KbName('2@') KbName('3#') KbName('4$')];
% key.demos       =   30:33;
key.demos   =   [KbName('1!') KbName('2@') KbName('3#') KbName('4$')];
% key.evals       =   30:36;

% obtain platform-independent responses
KbName('UnifyKeyNames');
KbQueueCreate; %creates cue using defaults
KbQueueStart; %starts the cue
% specifying response keys
key.topchoice = KbName('1!');
key.botchoice = KbName('2@');
%$key.dev_id = GetKeyboardIndices; % device id for checking responses.  -1 for all devices (but might be slow and miss some responses)
set_device_z;
% set valid keyboard response
keylookup = uint16(zeros([1 256]));
keylookup([key.topchoice key.botchoice]) = [0 1]; % 0 is top image and 1 is bot image
keylookup([KbName('1!') KbName('2@') KbName('3#') KbName('4$')]) = 0:3; % Numbers 1-7

%% Scantrigger detection (USE THIS FOR fMRI)
% Do nothing until the MR scantrigger (or keyboard button number "5") is detected
% % Screen('DrawText',w,'Waiting for start signal...',winRect(3)/2-60,winRect(4)/2); %present text
% % Screen('Flip', w);
% %
% % expStart = 0;
% % [keyIsDown, secs, keycode] = KbCheck(key.dev_id);                  % check response
% % %keep doing nothing if button "5" (scantrigger) isn't detected
% % while ~expStart
% %     [keyIsDown, start_scan, keycode] = KbCheck(key.dev_id);        % check response
% %     if keyIsDown
% %         disp(KbName(keycode));
% %     end
% %     if keycode(key.trigger)
% %         expStart = 1; %if button press
% %         time = GetSecs;
% %         tlog = cat(1,tlog, {'start',time});
% %     end
% % end

% Now restrict KbChecks to user responses
% expStart = 1; %if button press
% tlog = cat(1,tlog, {'start',GetSecs}); % Note: time log only logs start and stop (currently, easily extensible)
ListenChar(2);

%% Image presentation screen design
centpos    = [winRect(3)/2 winRect(4)/2 winRect(3)/2 winRect(4)/2];
imbox      = [-0.5*P.imsize(1), -0.5*P.imsize(2), 0.5*P.imsize(1), 0.5*P.imsize(2)];
pos1       = [-1.5*P.imsize(1)-P.Xgap, 0]; pos1 = [pos1 pos1];
    im1pos = centpos + pos1 + imbox;
pos2       = [-0.5*P.imsize(1), 0]; pos2 = [pos2 pos2];
    im2pos = centpos + pos2 + imbox;
pos3       = [0.5*P.imsize(1)+P.Xgap, -0.5*P.imsize(2)-P.Ygap]; pos3 = [pos3 pos3];
pos4       = [0.5*P.imsize(1)+P.Xgap,  0.5*P.imsize(2)+3*P.Ygap]; pos4 = [pos4 pos4];
    im3pos    = centpos + pos3 + imbox;
    im4pos    = centpos + pos4 + imbox;
pos34      = [0.5*P.imsize(1)+P.Xgap   0]; pos34 = [pos34 pos34];
    im34pos   = centpos + pos34 + imbox;
postxtmain = winRect(4)/2 - 0.5*P.imsize(2) - P.Ygap;
postxttop  = winRect(4)/2 - P.imsize(2) - 2*P.Ygap;
postxtbot  = winRect(4)/2 + 2*P.Ygap;
texty   = winRect(4)/2-(P.imsize(2) + 0.75*P.Xgap);

%$%$ might want to just move the demographics questions out of the scanner
%$ not



% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Phase 1 : ENCODING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%$%$ in the encoding practice trials, changed the presentation-wait time from the current 2 second to match later TR
%$%$ it's probably a good idea to add time logging to the practice trials as well?

%% Phase 1a : Encoding TRIAL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get the current time
time = GetSecs;
Screen('TextSize', w, P.fontsz);
% Print instructions
DrawFormattedText(w, P.instruc{1}, 'center', texty);
time = Screen('Flip', w, time);
time = time+3*P.readtime; %$ currently a 45 secs forced reading time
if ~fmri_autorun
  % After holding, forcing them to read, press any key to continue
  DrawFormattedText(w, P.instruc{1}, 'center', texty);
  DrawFormattedText(w, 'Press any key to continue...', 'center', winRect(4)/2+1.5*texty);
  Screen('Flip', w, time);
  RestrictKeysForKbCheck([key.nums]);
  KbWait(key.dev_id, 2);
  RestrictKeysForKbCheck([]);
  % Pause for a few seconds with a "+" blank on-screen
  DrawFormattedText(w, 'Practice trials.\n+', 'center', 'center'); %$%$ this shoul be "+"?
  time = Screen('Flip', w, time);
  time = time+3; % Pause just a bit before proceeding
  DrawFormattedText(w, 'Practice trials.\n+', 'center', 'center');
  DrawFormattedText(w, 'Press any key to continue...', 'center', winRect(4)/2+texty);
  time = Screen('Flip', w, time);
  KbWait(key.dev_id, 2);
end

for k = 1:3 % There are three trial sequences
    % 1 : Display text + images, one at a time
    % Create text
    Screen('TextSize', w, P.fontsz);
    % Get the bounding box of each chunk of text, so that we can center
    % each over its respective image
    [~, ~, textbnd0] = DrawFormattedText(w, sprintf('%s ',des_trial{2*(k-1)+1,1:2}), 'center', -100, [0.5 0.5 0.5]);
    [~, ~, textbnd1] = DrawFormattedText(w, sprintf('%s ',des_trial{2*(k-1)+1,3:4}), 'center', -100, [0.5 0.5 0.5]);

    % Create textures
    texture{1} = Screen('MakeTexture', w, imdata_trial{2*(k-1)+1,1});
    texture{2} = Screen('MakeTexture', w, imdata_trial{2*(k-1)+1,2});
    texture{3} = Screen('MakeTexture', w, imdata_trial{2*(k-1)+1,3});
    texture{4} = Screen('MakeTexture', w, imdata_trial{2*(k-1)+1+1,3});
    % Draw the textures, one by one
    txt12pos = (im1pos(1)+imbox(3))-(textbnd0(3)-textbnd0(1))/2;
    DrawFormattedText(w, sprintf('%s ',des_trial{2*(k-1)+1,1:2}), txt12pos, postxtmain);
    Screen('DrawTextures', w, texture{1}, [], im1pos);
    time = Screen('Flip', w, time,1);

    txt34pos = (im2pos(1)+imbox(3))-(textbnd1(3)-textbnd1(1))/2;
    DrawFormattedText(w, sprintf('%s ',des_trial{2*(k-1)+1,3:4}), txt34pos, postxtmain);
    Screen('DrawTextures', w, texture{2}, [], im2pos);
    time = Screen('Flip', w, time+P.trlength,1);

    txt56pos(1) = (im3pos(1)); %+imbox(3))-(textbnd2(3)-textbnd2(1))/2;
    txt56pos(2) = (im4pos(1)); %+imbox(3))-(textbnd3(3)-textbnd3(1))/2;
    DrawFormattedText(w, sprintf('%s ',des_trial{2*(k-1)+1,5:7}), txt56pos(1), postxttop);
    DrawFormattedText(w, sprintf('%s ',des_trial{2*(k-1)+1+1,5:7}), txt56pos(2), postxtbot);
    Screen('DrawTextures', w, texture{3}, [], im3pos);
    Screen('DrawTextures', w, texture{4}, [], im4pos);
    time = Screen('Flip', w, time+P.trlength); %$ clear the buffer this time

    % 2 : Keypress selection of 'preferred' scene
    %$%$ modified so the script will wait UNTIL A CHOICE IS MADE!
    %$%$ It's possible to make it such that a reminder is thrown after wait time elapsed - the only concern is that while the textures are being drawn
    %$%$ it's unclear if the computer can still do the routine key check thing, and we might not want to introduce that kind uncertainty to the RT data
    %$%$ curently this is addressed by redrawing the texture only AFTER the wait time has elapsed, before it's flipped
    %$%$ this is superior than having the textures drawn at the very beginning of the wait time
    %$%$ we can also set an additional longer wait time that ensures auto skip - but if that happens we probably have to throw away the person
    if fmri_autorun
      WaitSecs(1);
      lastchc = 0;
      keytime = GetSecs;
    else
      timedout = false;
      reminded = 0;
      rsp.RT = NaN; rsp.keyCode = []; rsp.keyName = [];
      while ~timedout
          [keyIsDown, keytime, keycode] = KbCheck(key.dev_id);
          if keycode(key.topchoice) || keycode(key.botchoice)
              % Choice made!
              lastchc = keylookup(keycode>eps);
              if numel(lastchc) == 1
                  break % IFF ONE BUTTON WAS PRESSED. Ignore if >1.
              end
          end
          if reminded == 0 && ((keytime - time) > P.encwaitforchoice) % only evaluates to true when the reminder text has not yet been drawn
              %% Choice omitted!
              %timedout = true;
              %lastchc = round(rand());

              DrawFormattedText(w, sprintf('%s ',des_trial{2*(k-1)+1,1:2}), txt12pos, postxtmain);
              DrawFormattedText(w, sprintf('%s ',des_trial{2*(k-1)+1,3:4}), txt34pos, postxtmain);
              DrawFormattedText(w, sprintf('%s ',des_trial{2*(k-1)+1,5:7}), txt56pos(1), postxttop);
              DrawFormattedText(w, sprintf('%s ',des_trial{2*(k-1)+1+1,5:7}), txt56pos(2), postxtbot);
              Screen('DrawTextures', w, texture{1}, [], im1pos);
              Screen('DrawTextures', w, texture{2}, [], im2pos);
              Screen('DrawTextures', w, texture{3}, [], im3pos);
              Screen('DrawTextures', w, texture{4}, [], im4pos);
              %$%$ reminder text, make sure to readjust the positioning
              DrawFormattedText(w, 'Please choose a scene', 'center',postxtmain - 2*P.Ygap);

              Screen('Flip', w);
              reminded = 1;
          end
      end
      KbReleaseWait;
    end


    %$%$ after a choice is made, remove the two scene images from the screen by re-flipping the objects
    %$%$ wait .75 seconds (half a TR, expected. later lock to the immediate next TR) before the chosen scene is flipped onto the screen
    %$%$ check if drawing time will become a problem in the TR lock phase (probably not. if the TR happens to overlap with drawing, worst case is waiting an extra TR)

    % 3 : New presentation of stimuli if not timed out
    %$ first remove the two scene images and only present the object images by redrawing and flipping the objects
    Screen('TextSize', w, 24);
    DrawFormattedText(w, sprintf('%s ',des_trial{2*(k-1)+1,1:2}), txt12pos, postxtmain);
    DrawFormattedText(w, sprintf('%s ',des_trial{2*(k-1)+1,3:4}), txt34pos, postxtmain);
    Screen('DrawTextures', w, texture{1}, [], im1pos);
    Screen('DrawTextures', w, texture{2}, [], im2pos);
    time = Screen('Flip', w, keytime,1);

    % Represent selected scenen
    DrawFormattedText(w, sprintf('%s ',des_trial{2*(k-1)+1+lastchc,5:7}), txt56pos(1+lastchc), postxtmain);
    texture{3} = Screen('MakeTexture', w, imdata_trial{2*(k-1)+1+lastchc,3});
    Screen('DrawTextures', w, texture{3}, [], im34pos);
    time = Screen('Flip', w, time + P.trlength/2); %$%$ added .75 sec wait time
    time = time + P.encholdingtime; %this variable is set in the cache script to be 3 sec, which already complies with the 2 tr plan

    %$%$ not including for the scanner
    % % 4 : Keypress selection of fluency
    % DrawFormattedText(w, strcat(P.fleval{:},'\n\n\n',P.flevalscale{:}), 'center', postxttop);
    % time = Screen('Flip', w, time);
    % timedout = false;
    % while ~timedout
    %     [keyIsDown, keytime, keycode] = KbCheck(key.dev_id);
    %     if any(keycode(key.evals))
    %         % Choice made!
    %         % (Nothing)
    %         break
    %     end
    %     if (keytime - time) > P.testwait
    %         % Choice omitted!
    %         timedout = true;
    %     end
    % end
    % KbReleaseWait;

    %$%$ fluency rating will be replaced with ITI after each trial
    % Pause for a few seconds with a "+" blank on-screen
    DrawFormattedText(w, '+', 'center', 'center'); %$%$ this shoul be "+"?
    time = Screen('Flip', w, time);
    time = time+3; % right now the iti value is hard coded at 6, but this could be moved to the cache script, and should be linked to tr in the next section

end


%% Phase 1b : Encoding -  run 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Screen('TextSize', w, 24);
% Pause for a few seconds with a "+" blank on-screen
% DrawFormattedText(w, '+', 'center', 'center');
% time = Screen('Flip', w, time);
% time = time+3; % Pause just a bit before proceeding
DrawFormattedText(w, 'Experiment will begin soon\n', 'center', 'center');
DrawFormattedText(w, 'Waiting for scanner warm-up...', 'center', winRect(4)/2+postxttop);
time = Screen('Flip', w, time);
%KbWait(key.dev_id, 2);

%$%$ once the scanner starts up wait for 2 triggers before starting the loops
if demoversion
    sct = 4;
else
    sct = numel(P.ENC);
end

%$%$ initialize timing data Vectors
rd.ititrtime = zeros(sct/2,1);
rd.ititrnum = zeros(sct/2,1);
rd.itionset = zeros(sct/2,1);
rd.itioffset = zeros(sct/2,1);
rd.im1trtime = zeros(sct/2,1);
rd.im1onset = zeros(sct/2,1);
rd.im1trnum = zeros(sct/2,1);
rd.im2trtime = zeros(sct/2,1);
rd.im2onset = zeros(sct/2,1);
rd.im2trnum = zeros(sct/2,1);
rd.im34trtime = zeros(sct/2,1);
rd.im34onset = zeros(sct/2,1);
rd.im34trnum = zeros(sct/2,1);
rd.reminded = zeros(sct/2,1);
rd.ooonset = zeros(sct/2,1);
rd.rescenetrtime = zeros(sct/2,1);
rd.resceneonset = zeros(sct/2,1);
rd.rescenetrnum = zeros(sct/2,1);
%$ additional run-level data storage, initialize data vectors
rd.object1 = cell(sct/2,1);
rd.object2 = cell(sct/2,1);
rd.scenecat = cell(sct/2,1);
rd.scenetop = cell(sct/2,1);
rd.scenebot = cell(sct/2,1);
rd.choice = zeros(sct/2,1)-1;
rd.rt = zeros(sct/2,1);
%$ additional storage vector. added 9/6/19 after adding randomization for which image is presented on top.
rd.stack = zeros(sct/2,1) - 1;

%$%$ once the scanner starts, wait 7 TR
DrawFormattedText(w, 'Experiment will begin soon\n', 'center', 'center');
DrawFormattedText(w, 'Waiting for scanner warm-up...', 'center', winRect(4)/2+postxttop);
[runTime, recorded, TRcounter] = WaitTRPulsePTB3_skyra(1); %$ runTime variabel represents the timing of the first trigger
firstonset = Screen('Flip', w, runTime);
%$
rd.firsttr = runTime;
rd.firstonset = firstonset;
%$ flip to ITI on TR 8
DrawFormattedText(w,'+','center','center');
[time,recorded,TRnum] = WaitTRPulsePTB3_skyra(7);
TRcounter = TRcounter + TRnum;
Screen('Flip', w, time);
%wait another TR to get itioffset: the last trigger of each iti
[itioffset, recorded, TRnum] = WaitTRPulsePTB3_skyra(1);
TRcounter = TRcounter + 1;
rd.firstoffset = itioffset;
%$ first stimulus on screen at TR 10

for k = 1:(sct/2)
    % 9/6/19 added randomizing which image gets presented on top
    % if the stackparam is 0, the order is as listed in the design matrix. elseif 1, they are flipped
    stackparam = randi(2) - 1;
    rd.stack(k) = stackparam;
    % 1 : Display text + images, one at a time
    % Create text
    Screen('TextSize', w, P.fontsz);
    % Get the bounding box of each chunk of text, so that we can center
    % each over its respective image
    [~, ~, textbnd0] = DrawFormattedText(w, sprintf('%s ',des{P.ENC(k),1:2}), 'center', -100, [0.5 0.5 0.5]);
    [~, ~, textbnd1] = DrawFormattedText(w, sprintf('%s ',des{P.ENC(k),3:4}), 'center', -100, [0.5 0.5 0.5]);

    % Create textures
    objidx = 1 + 4*floor((P.ENC(k)-1)/4);
    texture{1} = Screen('MakeTexture', w, imdata{objidx,1});
    texture{2} = Screen('MakeTexture', w, imdata{objidx,2});
    texture{3} = Screen('MakeTexture', w, imdata{P.ENC(k),3});
    texture{4} = Screen('MakeTexture', w, imdata{P.ENC(k)+1,3});
    % Draw the textures, one by one
    txt12pos = (im1pos(1)+imbox(3))-(textbnd0(3)-textbnd0(1))/2;
    DrawFormattedText(w, sprintf('%s ',des{P.ENC(k),1:2}), txt12pos, postxtmain);
    Screen('DrawTextures', w, texture{1}, [], im1pos);
    %$%$ once the textures are drawn, wait for the first trigger to flip
    [im1trtime, recorded, TRnum] = WaitTRPulsePTB3_skyra(1);
    im1onset = Screen('Flip', w, im1trtime,1);
    TRcounter = TRcounter + round((im1onset-itioffset)/P.TRlength); %$ this should always +1 to the counter unless the computer'sf'd up
    im1trnum = TRcounter;
    %$ time logging
    rd.im1trtime(k,1) = im1trtime;
    rd.im1onset(k,1) = im1onset;
    rd.im1trnum(k,1) = im1trnum;
    rd.object1{k,1} = des{P.ENC(k),2};
    %$%$ is this algorithm advisableor is there a better way to do it?
    %$%$ with thism, we record the actual onset time, and the ideal onset time would just be runTime + TRcounter*TRlength.

    txt34pos = (im2pos(1)+imbox(3))-(textbnd1(3)-textbnd1(1))/2;
    DrawFormattedText(w, sprintf('%s ',des{P.ENC(k),3:4}), txt34pos, postxtmain);
    Screen('DrawTextures', w, texture{2}, [], im2pos);
    %$%$ once the textures are drawn, wait for the first trigger to flip
    [im2trtime, recorded, TRnum] = WaitTRPulsePTB3_skyra(1);
    im2onset = Screen('Flip', w, im2trtime,1);
    TRcounter = TRcounter + round((im2onset-im1onset)/P.TRlength);
    im2trnum = TRcounter;
    %$ time logging
    rd.im2trtime(k,1) = im2trtime;
    rd.im2onset(k,1) = im2onset;
    rd.im2trnum(k,1) = im2trnum;
    rd.object2{k,1} = des{P.ENC(k),4};

    txt56pos(1) = (im3pos(1)); %+imbox(3))-(textbnd2(3)-textbnd2(1))/2;
    txt56pos(2) = (im4pos(1)); %+imbox(3))-(textbnd3(3)-textbnd3(1))/2;
    % draw scene options based on stack parameter
    if stackparam == 0
      DrawFormattedText(w, sprintf('%s ',des{P.ENC(k),5:7}), txt56pos(1), postxttop);
      DrawFormattedText(w, sprintf('%s ',des{P.ENC(k)+1,5:7}), txt56pos(2), postxtbot);
      Screen('DrawTextures', w, texture{3}, [], im3pos);
      Screen('DrawTextures', w, texture{4}, [], im4pos);
    elseif stackparam == 1
      DrawFormattedText(w, sprintf('%s ',des{P.ENC(k)+1,5:7}), txt56pos(1), postxttop);
      DrawFormattedText(w, sprintf('%s ',des{P.ENC(k),5:7}), txt56pos(2), postxtbot);
      Screen('DrawTextures', w, texture{4}, [], im3pos);
      Screen('DrawTextures', w, texture{3}, [], im4pos);
    end
    [im34trtime, recorded, TRnum] = WaitTRPulsePTB3_skyra(1);
    im34onset = Screen('Flip', w, im34trtime);
    TRcounter = TRcounter + round((im34onset-im2onset)/P.TRlength);
    im34trnum = TRcounter;
    %$ time logging
    rd.im34trtime(k,1) = im34trtime;
    rd.im34onset(k,1) = im34onset;
    rd.im34trnum(k,1) = im34trnum;
    rd.scenecat{k,1} = des{P.ENC(k),8};
    % store scene names conditioned on stackparam
    if stackparam ==0
      rd.scenetop{k} = des{P.ENC(k),6};
      rd.scenebot{k} = des{P.ENC(k)+1,6};
    elseif stackparam ==1
      rd.scenetop{k} = des{P.ENC(k)+1,6};
      rd.scenebot{k} = des{P.ENC(k),6};
    end

    % 2 : Keypress selection of 'preferred' scene
    if fmri_autorun
      randomrt = rand*3;
      WaitSecs(randomrt);
      lastchc = 0;
      keytime = GetSecs;
      P.chc(P.ENC(k)+lastchc,1) = 1; % marks which image is chosen
      % Store response info
      P.encRT(P.ENC(k)+lastchc)   = keytime - im34onset;
      % P.enckeyCode(P.ENC(k)+lastchc) = find(keycode);
      % P.enckeyName{P.ENC(k)+lastchc} = KbName(rsp.keyCode);
    else
      timedout = false;
      rsp.RT = NaN; rsp.keyCode = []; rsp.keyName = [];
      reminded = 0;
      while ~timedout
          [keyIsDown, keytime, keycode] = KbCheck(key.dev_id);
          if keycode(key.topchoice) || keycode(key.botchoice)
              % Choice made!
              lastchc = keylookup(keycode>eps);
              if numel(lastchc) == 1
                  % mapping from response to P file column storage is determined by stackparam - if stackparam == 1, mapping is reversed
                  if stackparam == 0
                    P.chc(P.ENC(k)+lastchc,1) = 1; % marks which image is chosen
                    % Store response info
                    P.encRT(P.ENC(k)+lastchc)   = keytime - im34onset;
                    P.enckeyCode(P.ENC(k)+lastchc) = find(keycode);
                    P.enckeyName{P.ENC(k)+lastchc} = KbName(rsp.keyCode);
                    break % IFF ONE BUTTON WAS PRESSED. Ignore if >1.
                  elseif stackparam == 1
                    P.chc(P.ENC(k)+1-lastchc,1) = 1; % marks which image is chosen
                    % Store response info
                    P.encRT(P.ENC(k)+1-lastchc)   = keytime - im34onset;
                    P.enckeyCode(P.ENC(k)+1-lastchc) = find(keycode);
                    P.enckeyName{P.ENC(k)+1-lastchc} = KbName(rsp.keyCode);
                    break % IFF ONE BUTTON WAS PRESSED. Ignore if >1.
                  end
              end
          end
          if reminded == 0 && (keytime - im34onset) > P.encwaitforchoice
            %% Choice omitted!
            %timedout = true;
            %lastchc = round(rand());

            %$ redraw texture along with a reminder text
            % draw scene options based on stack parameter
            DrawFormattedText(w, sprintf('%s ',des{P.ENC(k),1:2}), txt12pos, postxtmain);
            DrawFormattedText(w, sprintf('%s ',des{P.ENC(k),3:4}), txt34pos, postxtmain);
            Screen('DrawTextures', w, texture{1}, [], im1pos);
            Screen('DrawTextures', w, texture{2}, [], im2pos);
            if stackparam == 0
              DrawFormattedText(w, sprintf('%s ',des{P.ENC(k),5:7}), txt56pos(1), postxttop);
              DrawFormattedText(w, sprintf('%s ',des{P.ENC(k)+1,5:7}), txt56pos(2), postxtbot);
              Screen('DrawTextures', w, texture{3}, [], im3pos);
              Screen('DrawTextures', w, texture{4}, [], im4pos);
            elseif stackparam == 1
              DrawFormattedText(w, sprintf('%s ',des{P.ENC(k)+1,5:7}), txt56pos(1), postxttop);
              DrawFormattedText(w, sprintf('%s ',des{P.ENC(k),5:7}), txt56pos(2), postxtbot);
              Screen('DrawTextures', w, texture{4}, [], im3pos);
              Screen('DrawTextures', w, texture{3}, [], im4pos);
            end

            %$%$ reminder text, make sure to readjust the positioning
            DrawFormattedText(w, 'Please choose a scene', 'center',postxtmain - 2*P.Ygap);

            Screen('Flip', w);
            reminded = 1;
          end
      end
      KbReleaseWait;
      rd.reminded(k,1) = reminded;
      rd.choice(k,1) = lastchc;
      rd.rt(k,1) = keytime - im34onset;
    end


    % 3 : New presentation of stimuli %$%$ flip on the first trigger following the response
    %$%$ first make the two scene imnages go away by re drawing the object images and flip them as soon as a choice is made
    %$ depending on how long it takes to redraw the object images, this might manifest during the experiment as a lag between when the response is made and when the scene images disappear
    %$ ask lizzie about how to optimize this?  can't draw while waiting for response, but drawing prior to listening for response might introduce imprecision in RT measures
    Screen('TextSize', w, 24);
    DrawFormattedText(w, sprintf('%s ',des{P.ENC(k),1:2}), txt12pos, postxtmain);
    DrawFormattedText(w, sprintf('%s ',des{P.ENC(k),3:4}), txt34pos, postxtmain);
    Screen('DrawTextures', w, texture{1}, [], im1pos);
    Screen('DrawTextures', w, texture{2}, [], im2pos);
    ooonset = Screen('Flip', w, keytime,1); %$ only-object onset time, or object-object onset time :D
    rd.ooonset(k,1) = ooonset;

    % once the object images are on the screen, draw chosen scene, and then flip on the first trigger
    Screen('TextSize', w, 24);
    if stackparam == 0
      DrawFormattedText(w, sprintf('%s ',des{P.ENC(k)+lastchc,5:7}), txt56pos(1), postxtmain);
      texture{3} = Screen('MakeTexture', w, imdata{P.ENC(k)+lastchc,3});
      Screen('DrawTextures', w, texture{3}, [], im34pos);
    elseif stackparam == 1
      DrawFormattedText(w, sprintf('%s ',des{P.ENC(k)+1-lastchc,5:7}), txt56pos(1), postxtmain);
      texture{3} = Screen('MakeTexture', w, imdata{P.ENC(k)+1-lastchc,3});
      Screen('DrawTextures', w, texture{3}, [], im34pos);
    end
    [rescenetrtime, recorded, TRnum] = WaitTRPulsePTB3_skyra(1);
    resceneonset = Screen('Flip', w, rescenetrtime);
    TRcounter = TRcounter + round((resceneonset-im34onset)/P.TRlength);
    rescenetrnum = TRcounter;
    %$ time logging
    rd.rescenetrtime(k,1) = rescenetrtime;
    rd.resceneonset(k,1) = resceneonset;
    rd.rescenetrnum(k,1) = rescenetrnum;

    % time = Screen('Flip', w, keytime);
    % time = time + P.encholdingtime;

    % % 4 : Keypress selection of fluency
    % DrawFormattedText(w, strcat(P.fleval{:},'\n\n\n',P.flevalscale{:}), 'center', postxttop);
    % time = Screen('Flip', w, time);
    % timedout = false;
    % while ~timedout
    %     [keyIsDown, keytime, keycode] = KbCheck(key.dev_id);
    %     if sum(keycode(key.evals)) == 1 % IFF ONE BUTTON WAS PRESSED. Ignore if >1.
    %         % Choice made!
    %         P.ENCeval(P.ENC(k)+lastchc) = find(keycode>eps)-key.evals(1)+1;
    %         break
    %     end
    %     if (keytime - time) > P.testwait
    %         % Choice omitted!
    %         timedout = true;
    %     end
    % end
    % KbReleaseWait;

    %$%$ fluency rating will be replaced with ITI after each trial
    % Pause for a few seconds with a "+" blank on-screen
    DrawFormattedText(w, '+', 'center', 'center'); %$%$ this shoul be "+"?
    [ititrtime, recorded, TRnum] = WaitTRPulsePTB3_skyra(2);
    itionset = Screen('Flip', w, ititrtime);
    TRcounter = TRcounter + round((itionset-resceneonset)/P.TRlength);
    ititrnum = TRcounter;
    [itioffset, recorded, TRnum] = WaitTRPulsePTB3_skyra(1); %$ 4tr total - 3 + next loop begin + 1
    TRcounter = TRcounter + round((itioffset-itionset)/P.TRlength);
    %$ time logging
    rd.ititrtime(k,1) = ititrtime;
    rd.itionset(k,1) = itionset;
    rd.ititrnum(k,1) = ititrnum;
    rd.itioffset(k,1) = itioffset;
    %$ save relevant data from this run up to the end of the current trial]
    save([P.savedir '/' sprintf('%s_func_task-encoding_run-01.mat',P.ID)],'rd','P')
    disp("trial:" + string(k) ...
    + "  topscene:" + rd.scenetop{k} ...
    + "  botscene:" + rd.scenebot{k} ...
    + "  im1onset:" + string(im1onset-runTime) ...
    + "  im1tr:" + string(im1trnum) ...
    + "  im2onset:" + string(im2onset-runTime)...
    + "  im2tr:" + string(im2trnum)...
    + "  im34onset:" + string(im34onset-runTime)...
    + "  im34tr:" + string(im34trnum)...
    + "  choice:" + string(rd.choice(k,1))...
    + "  rt: " + string(rd.rt(k,1)))
end

%$ after the last run, additional TRs where cross hair is presented
WaitTRPulsePTB3_skyra(13);

%$%$ initiate the next run
stcbpe_selfenc_2
