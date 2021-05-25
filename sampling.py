

# with open('data/mc.csv') as reader, open('data/new_mc.csv', 'w') as writer:
#     for i, line in enumerate(reader):
#         if i % 100 == 0:
#             writer.write(line)


with open('data//original_data/friends.csv') as reader, open('data/test1/new_friends_17000.csv', 'w') as writer:
    for i, line in enumerate(reader):
        if i % 10 == 0:
            writer.write(line)


with open('data/votes_clean.csv') as reader, open('data/test1/new_votes.csv', 'w') as writer:
    for i, line in enumerate(reader):
        if i % 100 == 0:
            writer.write(line)
