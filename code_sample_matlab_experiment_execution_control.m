%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The executive script of the STCBPE study that runs both sessions
% upon calling the script prompts input for subject number and session number
% if session == 1, cary out randomization for the subject and store parameters
%                  run session 1 tasks: scene localizer, demographics,
%                  and familiarization
% if session == 2, first read in in stored randomization files from the first day
%                  use the retrieved files to run the mainwrap, and then the
%                  self-other localizer task
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% remember to modify stcbpe_mainwrap, such that the randomization is outside of that script, and change the wording for the mainwrap's initial prompting from "new experiment" to something more like "continuation after interruption?". also try to see if the "interrupted" variable can be set outside of the mainwrap, and just get passed
% this also requires that tthe taskcache script be split up, since not we want the demographics to be collected half way though tbut the image randomization be done at the beginning prior to the first task

% this script is to be run after T1w anatomical is done

%$ script parameters
interrupted = false;
once_interrupted = 0;
global demoversion; %$
demoversion = false; % True = only 3-5 of each phase, False = full expt
fmri_autorun = false; %$ if set to true, skip all responses

% solicit input - session 1 or session 2
n_sesh = input('Which session?\n1 = Session 1\n2 = Session 2\n');
while ~isnumeric(n_sesh)
  n_sesh = input('Which session?\n1 = Session 1\n2 = Session 2\n');
end

% solicit input - new experiment or previously interrupted session
interrupt_continue = input('Is this a new experiment or an continuing session?\n1 = New experiment\n2 = Continuing session\n');
while ~isnumeric(interrupt_continue)
  interrupt_continue = input('Is this a new experiment or an continuing session?\n1 = New experiment\n2 = Continuing session\n');
end

% based on input of session number, one branch of the conditional
if n_sesh == 1
  % based on whether resuming a broken session, one branch of the conditional
  if interrupt_continue == 1
    interrupted = false;
    % run randomization, load image data (for localizer AND familiarization)
    stcbpe_taskcache
    % start at the first scene localizer run.
    stcbpe_localizer_1
  else
    % solicit input - which session to resume on
    interrupted = true;
    % if it is not a new session, solicit input - which run to start on?
      % modify after making sure how many runs are in each of the tasks
    start_point = input('Which run?\n2 = scene_localizer_2\n3 = demographics\n4 = familiarization_1\n5 = familiarization_2\n6 = familiarization_3\n');
    while ~isnumeric(start_point)
      start_point = input('Which run?\n2 = scene_localizer_2\n3 = demographics\n4 = familiarization_1\n5 = familiarization_2\n6 = familiarization_3\n');
    end
    % solicit manually assigned subj Number
    % each of the run scripts after the first run detects "interrupted", and uses subid to retrieve the correct data file from the last run
    subject_code=input('Enter subject number: ');
    while ~isnumeric(subject_code)
      subject_code=input('Enter subject number: ');
    end
    subid = ['s',sprintf('%03d',subject_code)];
    % based on run number, call specific scripts
    if start_point == 2
      stcbpe_localizer_2
    elseif start_point == 3
      stcbpe_demog
    elseif start_point == 4
      stcbpe_familiarization_1
    elseif start_point == 5
      stcbpe_familiarization_2
    else
      stcbpe_familiarization_3
    end
  end
elseif n_sesh == 2
  % code in this portion is basically stcbpe_mainwrap
  if interrupt_continue == 1
      interrupted = false;
      % solicit manually assigned subj Number
      % each of the run scripts after the first run detects "interrupted", and uses subid to retrieve the correct data file from the last run
      subject_code=input('Enter subject number: ');
      while ~isnumeric(subject_code)
        subject_code=input('Enter subject number: ');
      end
      subid = ['s',sprintf('%03d',subject_code)];
      % first encoding run
      stcbpe_selfenc_1
  else
    interrupted = true;
    start_point = input('Which run?\n2 = self_enc_2\n3 = target_1\n4 = target_2\n5 = test_self_1\n6 = test_self_2\n7 = test_target\n8 = self_other_1\n9 = self_other_2\n');
    while ~isnumeric(start_point)
      start_point = input('Which run?\n2 = self_enc_2\n3 = target_1\n4 = target_2\n5 = test_self_1\n6 = test_self_2\n7 = test_target\n8 = self_other_1\n9 = self_other_2\n');
    end
    % solicit manually assigned subj Number
    % each of the run scripts after the first run detects "interrupted", and uses subid to retrieve the correct data file from the last run
    subject_code=input('Enter subject number: ');
    while ~isnumeric(subject_code)
      subject_code=input('Enter subject number: ');
    end
    subid = ['s',sprintf('%03d',subject_code)];
    % select run script based on input start_point
    if start_point == 2
      stcbpe_selfenc_2
    elseif start_point == 3
      stcbpe_target_1
    elseif start_point == 4
      stcbpe_target_2
    elseif start_point == 5
      stcbpe_test_self_1
    elseif start_point == 6
      stcbpe_test_self_2
    elseif start_point == 7
      stcbpe_test_target
    elseif start_point == 8
      stcbpe_selfother_1
    elseif start_point == 9
      stcbpe_selfother_2
    end

  end
end
