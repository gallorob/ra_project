'''
REQUIRED: terminalizer -> npm install -g terminalizer
'''
gif_dict = {'config': {
		    'command': 
		    'bash -l',
		    'cwd': None,
		    'env': {'recording': True}, 
		    'cols': 52, 
		    'rows': 19, 
		    'repeat': 0, 
		    'quality': 100, 
		    'frameDelay': 'auto', 
		    'maxIdleTime': 2000, 
		    'frameBox': {
		                 'type': 'solid', 
		                 'title': None, 
		                 'style': []
		                 }, 
		    'watermark': {
		                  'imagePath': None, 
		                  'style': {'position': 'absolute', 'right': '15px', 'bottom': '15px', 'width': '100px', 'opacity': 0.9}
		                 }, 
		    'cursorStyle': 'block', 
		    'fontFamily': 'Monaco, Lucida Console, Ubuntu Mono, Monospace', 
		    'fontSize': 12, 
		    'lineHeight': 1, 
		    'letterSpacing': 0, 
		    'theme': {
		              'background': 'transparent', 
		              'foreground': '#afafaf', 
		              'cursor': '#c7c7c7', 
		              'black': '#232628',  
		              'red': '#fc4384', 
		              'green': '#b3e33b', 
		              'yellow': '#ffa727', 
		              'blue': '#75dff2', 
		              'magenta': '#ae89fe', 
		              'cyan': '#708387', 
		              'white': '#d5d5d0', 
		              'brightBlack': '#626566', 
		              'brightRed': '#ff7fac', 
		              'brightGreen': '#c8ed71',  
		              'brightYellow': '#ebdf86', 
		              'brightBlue': '#75dff2', 
		              'brightMagenta': '#ae89fe', 
		              'brightCyan': '#b1c6ca', 
		              'brightWhite': '#f9f9f4'
		             }
		    }, 
		    'records': []
           }

from typing import Tuple

def from_log_to_dict(path: str) -> Tuple[dict, str, str]:
    '''
    Create a dict from the log file
    -----------
    Parameters:
        path: str,
            path to the log file
    -----------
    Returns:
        log_dict: dict,
            dict with episodes' names as keys and a list of relative steps as value; each step consists in a list of string,
        formula: str,
            string with the experiment's goal formula,
        reward_shaping: str,
            string with info about reward shaping.
    '''
    log_dict = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        f.close()

    # Extract the experiment's formula
    formula = lines[0][10:] 
    # Extract the reward shaping parameter
    if 'shaping' in lines[1]:
        reward_shaping = lines[1]
    else:  
        reward_shaping = ''
    # Set the simulation starting line
    starting_line_idx = 0
    while 'Episode' not in lines[starting_line_idx]:
        starting_line_idx += 1 
    lines = lines[starting_line_idx:]
    # Build the dict
    for line in lines:
        # Base case: Create a new key in the dict for each new Episode, ex: 'Episode 1':
        if 'Episode' in line and 'reward' not in line:
            ep = line.split(":")[0]
            log_dict[ep] = []
            step = []
        # Case 1: Append the complete step to the actual episode and empty the step list, ex: from 'Episode 1:' to 'Reward: -1.0'
        elif 'Episode' not in line and 'Reward' in line:
            step.append(line)
            if '+---------+\n' != step[0]: # To fix a bad formatting of log files caused by 'Episode n:+---------+'
                step.insert(0, '+---------+\n')
            log_dict[ep].append(step)
            step = []
        # Case 2: Append the actual line of the current step to the list containing all the lines of that step
        elif 'Episode' not in line and 'Reward' not in line:
            step.append(line)
        # Final Case: Append the ending line, ex: 'Episode ended; total reward: 10.0'
        else:
            log_dict[ep][-1].extend([line])

    return log_dict, formula, reward_shaping

def render(path: str, wait_time: float, episode: str, make_gif: bool, play_all: bool) -> None:
    '''
    Run a simulation of Taxi-v3 from a log file.
    -----------
    Parameters:
        path: str,
            path to the log file;
        wait_time: float,
            time between each simulation frame/step;
        episode: str, 
            index of the episode to run, if None runs all the episodes in the log file;
        make_gif: bool,
            if true saves a gif of the simulation;
        play_all: bool,
            if true plays all the episodes else only the solved ones.
    '''
    from time import sleep
    import yaml

    log_dict, formula, reward_shaping = from_log_to_dict(path)
    keys = []
    keys.append('Episode ' + episode) if episode != None else keys.extend([k for k in log_dict.keys()]) # Defines the episodes to run
    n_ep = 1 # Auxiliary episode index, needed for a better printing if failed episodes are skipped
    # For each episode
    for ep in keys:
        os.system('clear')
        # For each step in the episode
        for step in log_dict[ep]: 
            if not play_all and log_dict[ep][-1][-3].split('(')[1].split(')')[0] != 'Dropoff': # Consider only solved episodes
                n_ep -= 1
                break
            # Live printing of the episode title
            print(formula + reward_shaping + '\nEpisode: ', str(n_ep)+'\n')
            # Store the episode title for the gif
            record_step = chr(27) + "[3J" + chr(27) + "[H" + chr(27) + "[2J" + formula.split("\n")[0] + '\r\n' + reward_shaping + '\r\n \r\n' + 'Episode: ' + str(n_ep) + '\r\n\r\n'
            # For each line in the current step
            for table_line in step:
                # Handle the formatting of current episode's last line
                if 'ended' in table_line:
                    print('\n        ' + table_line, end='\r') # live printing
                    record_step += '\r\n\r\n        ' + table_line.split("\n")[0] + '\r\n\r' # store line for the gif
                # Handle the formatting of current episode's generic line
                else:
                    print('\t\t  ' + table_line, end='\r') # live printing
                    record_step += '\t\t  ' + table_line.split("\n")[0] + '\r\n\r' # store line for the gif
            # Append the current step to the dict with all the steps for the gif
            # If the step is the last one, increment the frame's length before switching to the next frame
            if step == log_dict[ep][-1]:
                gif_dict['records'].append({'delay': wait_time*1000, 'content': record_step})
                gif_dict['records'].append({'delay': 1000, 'content': record_step})
                sleep(1.5) # default waiting time at the end of the episode
            else:
                gif_dict['records'].append({'delay': wait_time*1000, 'content': record_step})
                sleep(wait_time) # waiting time between each frame
            os.system('clear')
        n_ep += 1

    # Either save the final gif or not
    if make_gif:
        print("Making a cool gif")
        gif_dst = path.split(path.split('/')[-1])[0] + path.split('/')[-1].split('.')[0]
        print('Destintion: ', gif_dst + '.gif')
        with open(gif_dst + '.yml', 'w') as outfile:
            yaml.dump(gif_dict, outfile, default_flow_style=False)
        os.system('terminalizer render ' + gif_dst + ' -o ' + gif_dst)
        os.system('rm ' + gif_dst + '.yml')

if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser(description='Taxi v3 log visualizer')
    parser.add_argument('--p', type=str, dest='log_path',
                        help='Path to .log file')
    parser.add_argument('--t', type=float, dest='step_wait', default=0.5,
                        help='Optional; Seconds to wait from one step to the next one')
    parser.add_argument('--e', type=str, dest='episode', default=None,
                        help='Optional; Specific Episode to visualize')
    parser.add_argument('--g', type=bool, dest='make_gif', default=False,
                        help='Optional; True to render a gif')
    parser.add_argument('--all', type=bool, dest='play_all', default=False,
                        help='Optional; Plays either all the episodes if True or only the solved ones if False')
    args = parser.parse_args()
    try:
        # Handle path to directory
        if os.path.isdir(args.log_path):
            for filename in [f for f in os.listdir(args.log_path) if f.split('.')[-1] == 'log']:
                render(path=args.log_path + filename,
                        wait_time=args.step_wait,
                        episode=args.episode,
                        make_gif=args.make_gif,
                        play_all=args.play_all)
        # Handle path to log file
        else:
            render(path=args.log_path,
                    wait_time=args.step_wait,
                    episode=args.episode,
                    make_gif=args.make_gif,
                    play_all=args.play_all)
    except FileNotFoundError:
        print('\033[91m[ERROR] No such file or directory')

