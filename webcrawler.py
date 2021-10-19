# -*- coding: utf-8 -*-
"""
This file contains all the functions needed for webcrawling NBA data
"""

import urllib.request
import os
import csv
from bs4 import BeautifulSoup
import numpy as np
import pickle
import dateutil.parser as parser
import datetime

# The alphabet used for player names (NOTE NO 'X'!)
ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
            'q', 'r', 's', 't', 'u', 'v', 'w', 'y', 'z']
# Default to the last 5 years of data
MIN_YEAR_DEFAULT = datetime.datetime.now().year-5
# URLS
URL_TEAMS = 'https://www.basketball-reference.com/teams/'
URL_PLAYERS = 'https://www.basketball-reference.com/players/'
URL_BASE = 'https://www.basketball-reference.com'

# Shot range (in feet) bin thresholds
SHOT_LIMITS = [3, 10, 16, 22]
PLAYER_HEADER = ['date_game',
                 'team_id',
                 'opp_id',
                 'game_location',
                 'gs',
                 'mp',
                 'fg',
                 'fga',
                 'fg2',
                 'fg2a',
                 'fg3',
                 'fg3a',
                 'ft',
                 'fta',
                 'orb',
                 'drb',
                 'trb',
                 'ast',
                 'stl',
                 'blk',
                 'tov',
                 'pf',
                 'pts',
                 'plus_minus']
OPP_HEADER = ['fg',
              'fga',
              'fg3',
              'fg3a',
              'ft',
              'fta',
              'orb',
              'drb',
              'trb',
              'ast',
              'stl',
              'blk',
              'tov',
              'pf',
              'pts',
              'opp_fg',
              'opp_fga',
              'opp_fg3',
              'opp_fg3a',
              'opp_ft',
              'opp_fta',
              'opp_orb',
              'opp_drb',
              'opp_trb',
              'opp_ast',
              'opp_stl',
              'opp_blk',
              'opp_tov',
              'opp_pf',
              'opp_pts']


def get_soup(url):
    connection = urllib.request.urlopen(url)
    soup = BeautifulSoup(connection, 'html.parser')
    return soup


def write_csv(filename, header, data, append=False):
    if append:
        data_to_add = []
        # Read in the file to get the latest date
        with open(filename, 'r') as csv_file:
            reader = csv.reader(csv_file)
            player_data = []
            for row in reader:
                player_data.append(row)
        # The last game datetime
        header_csv = player_data[0]
        last_game = player_data[-1]
        last_game_datetime = parser.parse(last_game[header_csv.index('date_game')])
        # Find the data where the datetime is greater than the current one
        found_new_game = False
        if len(data) > 0:
            for index in range(len(data)):
                if parser.parse(data[index][header.index('date_game')]) > last_game_datetime:
                    found_new_game = True
                    break
            if found_new_game:
                data_to_add = data[index:]
            else:
                data_to_add = []
        data = player_data[1:]+data_to_add
    with open(filename, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(header)
        for dataLine in data:
            csv_writer.writerow(dataLine)


def does_playerfile_exist(player_name, csv_dir):
    # Go through all player files and see if the name is in any of them
    player_files = [f for f in os.listdir(csv_dir) if os.path.isfile(os.path.join(csv_dir, f))]
    for player_file in player_files:
        if player_name in player_file:
            return True
    return False


def get_opp_stats(opp_link, game_datetime, opp_header=OPP_HEADER, verbose=False,
                  opp_dict={}):
    if opp_link in opp_dict.keys():
        header = opp_dict[opp_link][0]
        opp_stats = opp_dict[opp_link][1]
    else:
        soup = get_soup(opp_link)
        if verbose:
            print('Connected to: ' + opp_link)
        '''
        GET THE TABLE
        '''
        opp_table = soup.find('table', attrs={'id': 'tgl_basic'})
        opp_stats = []
        header = []
        got_header = False
        rows = opp_table.findAll('tr')
        for row in rows:
            # See if it is a header
            if len(row.findAll('th')) > 10 and (not got_header):
                # Get the header
                header = get_header_from_row(row)
                got_header = True
            new_col = []
            cells = row.findAll('td')
            if len(row.findAll('th')) < 2:
                for cell in cells:
                    if 'csk' in cell.attrs.keys():
                        text = cell['csk']
                    else:
                        text = cell.text.replace('&nbsp;', '')
                    new_col.append(text)
            if len(new_col) > 5:
                opp_stats.append(new_col)
        opp_dict[opp_link] = [header, opp_stats]
    '''
    Now we have the table (opp_stats). Get the info we want (opp_header features).
    '''
    # Go through the features we want for each game
    opp_feature_vectors = []
    done_with_games = False
    for game in opp_stats:
        feature_vector = []
        for label in opp_header:
            if parser.parse(game[header.index('date_game')]) < game_datetime:
                feature_vector.append(get_features_from_row(game, header, label))
            else:
                done_with_games = True
                break
        if done_with_games:
            break
        else:
            opp_feature_vectors.append(feature_vector)
    num_games = len(opp_feature_vectors)
    opp_features = []
    if num_games > 0:
        # Loop through each feature
        for stat_index in range(len(opp_header)):
            opp_stat_to_avg = []
            for dummy_vector in opp_feature_vectors:
                opp_stat_to_avg.append(float(dummy_vector[stat_index]))
            opp_features.append(np.mean(opp_stat_to_avg))
    return opp_header, opp_features


def get_number_from_table(row, header, label):
    try:
        # Sometimes the row will have less elements than header if the player did not play that game
        if header.index(label) >= len(row):
            value = 0
        elif len(row[header.index(label)]) > 0:
            value = float(row[header.index(label)])
        else:
            value = 0
    except Exception as e:
        print(e)
        value = 0
    return value


def get_active_player_list_from_letter(letter, url_players=URL_PLAYERS,
                                       url_base=URL_BASE, verbose=False,
                                       min_year=MIN_YEAR_DEFAULT):
    player_links = []
    player_names = []
    url_player_list = url_players+letter
    soup = get_soup(url_player_list)
    table = soup.find('table', attrs={'id': 'players'})
    rows = table.findAll('tr')
    # Go through the rows and see if there are any active players (ignore the header)
    for row in rows[1:]:
        cells = row.findAll('td')
        if len(row.findAll('strong')) > 0 or int(cells[1].text) >= min_year:
            # This is an active player! Get the html link and player name
            player_links.append(url_base+row.find('th').find('a')['href'])
            player_names.append(str(row.find('th').text))
            if verbose:
                print('Found ' + str(row.find('th').text) + "'s link!")
    return player_links, player_names


def get_features_from_row(game, header, label):
    if label == 'game_location':
        if game[header.index(label)] == '@':
            value = 1
        else:
            value = 0
    elif label == 'fg2':
        fg_tot = get_number_from_table(game, header, 'fg')
        fg3 = get_number_from_table(game, header, 'fg3')
        value = fg_tot-fg3
    elif label == 'fg2a':
        fga_tot = get_number_from_table(game, header, 'fga')
        fg3a = get_number_from_table(game, header, 'fg3a')
        value = fga_tot-fg3a
    elif ('drb' not in header) and label == 'drb':
        rb_tot = get_number_from_table(game, header, 'trb')
        orb = get_number_from_table(game, header, 'orb')
        value = rb_tot-orb
    elif ('opp_drb' not in header) and label == 'opp_drb':
        rb_tot = get_number_from_table(game, header, 'opp_trb')
        orb = get_number_from_table(game, header, 'opp_orb')
        value = rb_tot-orb
    elif label == 'date_game' or label == 'opp_id' or label == 'team_id':
        value = game[header.index(label)]
    else:
        value = get_number_from_table(game, header, label)
    return value


def get_theta(x, y):
    x_shift = x-240
    y_shift = y-24
    theta = np.arctan2(y_shift, x_shift)
    return theta


def get_shots_from_game(url):
    shot_dictionary = {}
    soup = get_soup(url)
    images = soup.findAll('img', attrs={'alt': 'nbahalfcourt'})
    for image in images:
        wrapper = image.parent
        team_id = wrapper['id'][-3:]
        team_shots = []
        # Get the shooting stats
        shots = wrapper.findAll('div')
        for shot in shots:
            # Get the shot info
            position_text = shot['style']
            shot_text = shot.attrs['tip']
            shot_text = shot_text[shot_text.index('>')+1:shot_text.rindex('<')]
            distance = int(shot_text[shot_text.index('from')+4:shot_text.rindex('ft')])
            y = int(position_text[position_text.index('top')+4:position_text.index('px;left:')])
            x = int(position_text[position_text.index('left:')+5:position_text.rindex('px;')])
            theta = get_theta(x, y)
            if 'missed' in shot_text:
                made = 0
                player_name = shot_text[0:shot_text.index('missed')-1]
            else:
                made = 1
                player_name = shot_text[0:shot_text.index('made')-1]
            team_shots.append([player_name, made, distance, theta])
        shot_dictionary[team_id] = team_shots
    return shot_dictionary


def get_shot_matrix(url, shot_dict={}, verbose=False):
    if url in shot_dict.keys():
        shot_matrix = shot_dict[url]
    else:
        shot_matrix = get_shots_from_game(url)
        shot_dict[url] = shot_matrix
        if verbose:
            print('Got shot data from : ' + url)
    return shot_matrix


def get_shot_index(shot, shot_limits):
    index = 0
    for i in range(len(shot_limits)):
        if shot > shot_limits[i]:
            index += 1
    return index


def get_shot_stats(shot_matrix, player_name, shot_limits=SHOT_LIMITS):
    shot_teams = shot_matrix.keys()
    missed_shots = np.zeros(len(shot_limits)+1, dtype=int)
    made_shots = np.zeros(len(shot_limits)+1, dtype=int)
    for team in shot_teams:
        for shot in shot_matrix[team]:
            if player_name == shot[0]:
                if shot[1] == 1:
                    made_shots[get_shot_index(shot[2], shot_limits)] += 1
                else:
                    missed_shots[get_shot_index(shot[2], shot_limits)] += 1
    shot_stats = list(np.hstack((made_shots, missed_shots)))
    shot_header = []
    shot_header_dummy = []
    for shot_lim in shot_limits:
        shot_header_dummy.append('range'+str(shot_lim))
    shot_header_dummy.append('range'+str(shot_lim)+'+')
    for prefix in ['made_', 'missed_']:
        for label in shot_header_dummy:
            shot_header.append(prefix+label)
    return shot_header, shot_stats


def get_header_from_row(row):
    header = []
    labels = row.findAll('th')
    for cell in labels:
        # Ignore the ranker header
        if cell['data-stat'] == 'ranker':
            continue
        header.append(cell['data-stat'])
    return header


def get_player_info_fast(url, player_name, player_header=PLAYER_HEADER,
                         opp_header=OPP_HEADER, opp_dict={}, shot_dict={},
                         verbose=False):
    soup = get_soup(url)
    if verbose:
        print('Connected to : ' + url)
    '''
    GET THE TABLE FROM THE PLAYER'S GAMELOG PAGE
    '''
    # Get the table for the player
    table = soup.find('table', attrs={'id': 'pgl_basic'})
    player_stats = []
    header = []
    game_links = []
    got_header = False
    rows = table.findAll('tr')
    for row in rows:
        cells = row.findAll('td')
        # See if it is a header
        if len(row.findAll('th')) > 10 and (not got_header):
            # Get the header
            header = get_header_from_row(row)
            got_header = True
        newcol = []
        if len(row.findAll('th')) < 2:
            for cell in cells:
                if len(cell.findAll('a')) == 1 and cell.a['href'][1:10] == 'boxscores':
                    game_links.append(cell.a['href'])
                if 'csk' in cell.attrs.keys():
                    text = cell['csk']
                else:
                    text = cell.text.replace('&nbsp;', '')
                newcol.append(text)
        if len(newcol) > 5:
            player_stats.append(newcol)
    '''
    GET THE PLAYER_HEADER FEATURES ONLY
    '''
    player_stats_filtered = []
    for row in player_stats:
        filtered_row = []
        for label in player_header:
            filtered_row.append(get_features_from_row(row, header, label))
        player_stats_filtered.append(filtered_row)
    '''
    GET THE OPPOSING TEAM'S STATS FOR EACH GAME
    '''
    # Go through the table again and get the opposing team links
    opp_index = header.index('opp_id')
    shot_index = header.index('date_game')
    opp_links = []
    shot_links = []
    for row in rows:
        cells = row.findAll('td')
        if len(row.findAll('th')) < 2:
            opp_links.append(cells[opp_index].a['href'])
            boxscore_link = cells[shot_index].a['href']
            shot_link = boxscore_link[0:boxscore_link.rindex('/')+1] + 'shot-chart' + \
                        boxscore_link[boxscore_link.rindex('/'):]
            shot_links.append(URL_BASE + shot_link)
    # Now we have the links and the game dates, get the opponent average stats so far
    date_index = header.index('date_game')
    net_opp_stats = []
    net_shot_stats = []
    for game_index in range(len(player_stats)):
        game_datetime = parser.parse(player_stats[game_index][date_index])
        # Get the opponent's stats for the game
        opp_header, opp_stats = get_opp_stats(URL_BASE+opp_links[game_index][0:-5]+'/gamelog',
                                              game_datetime, opp_header=opp_header,
                                              verbose=verbose, opp_dict=opp_dict)
        # Get the player's shot stats for the game if needed
        shot_matrix = get_shot_matrix(shot_links[game_index], shot_dict=shot_dict, verbose=verbose)
        shot_header, shot_stats = get_shot_stats(shot_matrix, player_name)
        net_shot_stats.append(shot_stats)
        net_opp_stats.append(opp_stats)
    '''
    GET THE INJURIES (IF POSSIBLE IN THE FUTURE)
    FUTURE WORK!!!!!!!!!!!!
    '''
    amended_opp_header = ['opp_'+opp_label for opp_label in opp_header]
    net_header = player_header+shot_header+amended_opp_header
    net_stats = [player_stats_filtered[i]+net_shot_stats[i] + net_opp_stats[i] \
                 for i in range(len(player_stats_filtered))]
    return net_header, net_stats


def get_active_player_list(alphabet=ALPHABET, verbose=False,
                           active_player_list_filename=None,
                           min_year=MIN_YEAR_DEFAULT):
    if active_player_list_filename is None:
        # Set the active_player_list_filename
        active_player_list_filename = 'active_player_links.pickle'
    if os.path.isfile(active_player_list_filename):
        with open(active_player_list_filename, 'rb') as f:
            player_links, player_names = pickle.load(f)
    else:
        player_links = []
        player_names = []
        for letter in alphabet:
            # Get a url list of every player whose name begins with the letter
            player_links_dummy, player_names_dummy = get_active_player_list_from_letter(letter, verbose=verbose,
                                                                                        min_year=min_year)
            player_links = player_links+player_links_dummy
            player_names = player_names+player_names_dummy
            if verbose:
                print('Done with ' + letter + ' names!')
        # Save the data
        with open(active_player_list_filename, 'wb') as f:
            pickle.dump([player_links, player_names], f)
    return player_links, player_names


def get_active_player_data(min_year=MIN_YEAR_DEFAULT, csv_dir=None, verbose=False,
                           skip_if_exists=True, save_data=False,
                           use_active_player_list=False,
                           update_latest_games=False,
                           opp_dict_filename=None,
                           shot_dict_filename=None):
    # Make sure we have a csv dir
    if csv_dir is None:
        print('Please input a csv directory!!')
        return 0

    # Make an opponent dictionary to story values for speed
    if opp_dict_filename is not None:
        with open(opp_dict_filename) as f:
            opp_dict = pickle.load(f)
    else:
        opp_dict = {}
    if shot_dict_filename is not None:
        with open(shot_dict_filename) as f:
            shot_dict = pickle.load(f)
    else:
        shot_dict = {}
    # Get a list of every active player (link and name)
    player_links, player_names = get_active_player_list(verbose=verbose,
                                                        min_year=min_year)
    # Go through each player and get individual game stats
    for player_index in range(len(player_links)):
        player_link_base = player_links[player_index]
        player_name = player_names[player_index]
        # See if we already have the player's data
        if not (skip_if_exists and does_playerfile_exist(player_name, csv_dir=csv_dir)):
            # We should get the player's data
            player_data = []
            for year in np.linspace(min_year, datetime.datetime.now().year, datetime.datetime.now().year-min_year+1):
                player_link = player_link_base[:-5]+'/gamelog/'+str(int(year))+'/'
                try:
                    net_header, net_stats = get_player_info_fast(player_link, player_name,
                                                                 player_header=PLAYER_HEADER,
                                                                 opp_dict=opp_dict, shot_dict=shot_dict,
                                                                 opp_header=OPP_HEADER, verbose=verbose)
                    player_data = player_data + net_stats
                except Exception as e:
                    print(e)
                    if verbose:
                        print('No data for ' + player_name + "'s %d season" % year)
                if verbose:
                    print('Done with ' + player_name + "'s %d season." % year)
            # Save the data
            if update_latest_games and len(player_data) > 1:
                write_csv(os.path.join(csv_dir, player_name+'.csv'),
                          net_header, player_data, append=True)
            elif not update_latest_games:
                write_csv(os.path.join(csv_dir,player_name+'.csv'),
                          net_header, player_data)
            if verbose:
                print('Saved the data in ' + os.path.join(csv_dir,player_name+'.csv'))
    # Save the opp dictionary and shot dictionary
    with open(opp_dict_filename,'w') as f:
        pickle.dump(opp_dict, f)
    with open(shot_dict_filename,'w') as f:
        pickle.dump(shot_dict, f)


if __name__ == '__main__':
    # Call the webcrawler to find all player data from after a certain year
    get_active_player_data(min_year=MIN_YEAR_DEFAULT, csv_dir='player_data', verbose=True)
