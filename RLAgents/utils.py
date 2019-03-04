from datetime import datetime


def discount(rewards, gamma):
    R = 0
    G = []
    for r in rewards[::-1]:
        R = r + gamma * R
        G = [R] + G
    return G


def get_cool_looking_datestring():
    now = datetime.now()
    return '' + str(now.day) + '_' + str(now.month) + '_' + str(now.hour) + ':' + str(now.minute) + ':' + str(now.second)    