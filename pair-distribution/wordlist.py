# ~100 most common used words + 50 most used nouns
words = "a, about, after, all, also, an, and, any, are, as, at,\
 away, be, because, been, before, being, between, both, but,\
 by, came, can, come, could, day, did, do, down, each, even,\
 first, for, from, get, give, go, had, has, have, he, her,\
 here, him, his, how, I, if, in, into, is, it, its, just,\
 know, like, little, long, look, made, make, many, may, me,\
 might, more, most, much, must, my, never, new, no, not,\
 now, of, on, one, only, or, other, our, out, over, people,\
 say, see, she, should, so, some, take, tell, than, that,\
 the, their, them, then, there, these, they, thing, think,\
 this, those, time, to, two, up, us, use, very, want,\
 was, way, we, well, were, what, when, where, which,\
 while, who, will, with, would, year, you, your, Time,\
 Year, People, Way, Day, Man, Thing, Woman, Life, Child,\
 World, School, State, Family, Student, Group, Country,\
 Problem, Hand, Part, Place, Case, Week, Company, System,\
 Program, Question, Work, Government, Number, Night,\
 Point, Home, Water, Room, Mother, Area, Money, Story, Fact,\
 Month, Lot, Right, Study, Book, Eye, Job, Word, Business, Service"

words = words.split(", ")
words = [i.lower() for i in words]
index_to_word = {i:words[i] for i in range(len(words))}
word_to_index = {index_to_word[list(index_to_word.keys())[i]]:list(index_to_word.keys())[i] for i in range(len(words))}