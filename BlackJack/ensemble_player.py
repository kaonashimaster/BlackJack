import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import sys
sys.path.append(os.path.abspath('.'))
import argparse
import socket
import torch
import numpy as np
from classes import Action, Player, get_action_name
from config import PORT, BET, INITIAL_MONEY, N_DECKS
from NN_structure import BJNet
from classes import QTable

MODEL_PAD_DIM = 17

# Map Action enum to command string used by dealer
ACTION_TO_CMD = {
    Action.HIT: 'hit',
    Action.STAND: 'stand',
    Action.DOUBLE_DOWN: 'double_down',
    Action.SURRENDER: 'surrender',
    Action.RETRY: 'retry'
}


def build_nn_input(player: Player):
    """Build a 17-dim input for BJNet from current player state.
    Format: [score, n_cards, soft_flag, dealer_open_score] + 13 zeros (card probs)
    """
    score = player.get_score()
    n_cards = player.get_num_player_cards()
    # Best-effort: detect soft hand
    soft = 0.0
    # dealer open card score
    dealer_score = 0.0
    if len(player.dealer_hand.cards) > 0:
        dc = player.dealer_hand.cards[0]
        dealer_score = min(10, (dc % 13) + 1)
    vec = [float(score), float(n_cards), float(soft), float(dealer_score)] + [0.0]*13
    arr = np.asarray([vec], dtype=np.float32)
    return torch.from_numpy(arr)


def select_action_ensemble(state, q_table: QTable, nn_model, device):
    # state is (score,length)
    if q_table is not None:
        # check if any Q exists for this state
        # QTable.table keys are (state, action)
        # we check presence by scanning keys
        found = False
        for (s,a) in q_table.table.keys():
            if s == state:
                found = True
                break
        if found:
            return q_table.get_best_action(state)
    # fallback to NN if present
    if nn_model is not None:
        # caller must have set player global; we'll handle building input elsewhere
        return None
    # final fallback: conservative STAND
    return Action.STAND


def main():
    parser = argparse.ArgumentParser(description='Ensemble AI player (Q table + NN)')
    parser.add_argument('--games', type=int, default=1000, help='num. of games to play')
    parser.add_argument('--qtable', type=str, default='', help='Q table file to load (pickle)')
    parser.add_argument('--model', type=str, default='', help='NN model path (pth)')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU id; negative for CPU')
    args = parser.parse_args()

    device = 'cpu'
    if torch.cuda.is_available() and args.gpu >= 0:
        device = f'cuda:{args.gpu}'

    # prepare player
    player = Player(initial_money=INITIAL_MONEY, basic_bet=BET)

    # load q_table if provided
    q_table = None
    if args.qtable != '':
        try:
            import pickle
            with open(args.qtable, 'rb') as f:
                table = pickle.load(f)
            q_table = QTable(Action, default_value=0)
            q_table.table = table
            print('Loaded Q table:', args.qtable)
        except Exception as e:
            print('Failed to load Q table:', e)
            q_table = None

    # load nn model if provided
    nn_model = None
    if args.model != '':
        try:
            nn_model = BJNet()
            nn_model.load_state_dict(torch.load(args.model, map_location=device))
            nn_model.to(device)
            nn_model.eval()
            print('Loaded NN model:', args.model)
        except Exception as e:
            print('Failed to load NN model:', e)
            nn_model = None

    total_games = args.games
    for gid in range(1, total_games+1):
        # connect to dealer
        soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        soc.connect((socket.gethostname(), PORT))
        soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # bet
        bet, money = player.set_bet()
        # receive shuffle status
        cardset_shuffled = player.receive_card_shuffle_status(soc)
        if cardset_shuffled:
            # if NN expects card counter it is not implemented here
            pass
        # receive initial cards
        dc, pc1, pc2 = player.receive_init_cards(soc)

        # game loop
        while True:
            state = (player.get_score(), player.get_num_player_cards())
            # try q-table
            action = None
            if q_table is not None:
                # check presence
                found = any(s == state for (s,a) in q_table.table.keys())
                if found:
                    action = q_table.get_best_action(state)
            if action is None and nn_model is not None:
                x = build_nn_input(player).to(device)
                with torch.no_grad():
                    y = nn_model(x)
                    probs = torch.softmax(y, dim=1).cpu().numpy()[0]
                # mapping action_set order must match BJNet output
                action_set = [Action.DOUBLE_DOWN, Action.HIT, Action.RETRY, Action.STAND, Action.SURRENDER]
                idx = int(np.argmax(probs))
                action = action_set[idx]
            if action is None:
                action = Action.STAND

            # send command
            cmd = ACTION_TO_CMD[action]
            player.send_message(soc, cmd)

            # receive
            if action == Action.HIT:
                pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True)
            elif action == Action.DOUBLE_DOWN:
                pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True)
            elif action == Action.RETRY:
                pc, score, status, rate, dc = player.receive_message(dsoc=soc, get_player_card=True, get_dealer_cards=True, retry_mode=True)
            else:
                score, status, rate, dc = player.receive_message(dsoc=soc, get_dealer_cards=True)

            if status != 'unsettled':
                # update money
                player.update_money(rate=rate)
                soc.close()
                break
        # end of one game
    # all games done
    print('Final money after', total_games, 'games:', player.get_money())

if __name__ == '__main__':
    main()
