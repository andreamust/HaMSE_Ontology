# DISCLAIMER ABOUT THE CODE
###############################
# Andrea Poltronieri 2020

import numpy as np
from vis.models.indexed_piece import Importer
import music21
from music21 import instrument, corpus
import csv

DATASET_PATH = ""
OUTPUT_PATH = ""

score_1 = "/Users/andrea/Documents/DHDK/Thesis/Ontology Data/BWV_1.6.mxl"
score_2 = corpus.getWork('bwv1.6')

def get_metadata(music_score):
    song = music21.converter.parse(music_score)
    undefined = 'undefined'
    if song.metadata.composers:
        composers = ('composers', song.metadata.composers)
    elif not song.metadata.composers:
        composers = ('composers', undefined)
    if song.metadata.title:
        title = ('title', song.metadata.composers)
    elif not song.metadata.title:
        title = ('title', undefined)
    if song.metadata.alternativeTitle:
        alternativeTitle = ('alternativeTitle', song.metadata.alternativeTitle)
    elif not song.metadata.alternativeTitle:
        alternativeTitle = ('alternativeTitle', undefined)
    if song.metadata.movementName:
        movementName = ('movementName', song.metadata.movementName)
    elif not song.metadata.movementName:
        movementName = ('movementName', undefined)
    if song.metadata.movementNumber:
        movementNumber = ('movementNumber', song.metadata.movementNumber)
    elif not song.metadata.movementNumber:
        movementNumber = ('movementNumber', undefined)
    all_meta_list = list(song.metadata.all())
    for x in all_meta_list:
        for y in x:
            if y == 'arranger':
                arranger = ('arranger', (x[1]))
            else:
                arranger = ('arranger', undefined)

    metadata_list = [composers, title, alternativeTitle, movementName, movementNumber, arranger]
    print(metadata_list)


def occurrences_sum(list_num, index):
    result_val = 0
    for val in list(list_num.values[index]):
        if np.isnan(val):
            val = 0
        result_val += val
    return int(result_val)


def music_key(music_score):
    song = music21.converter.parse(music_score)
    parts = song.getElementsByClass(music21.stream.Part)
    measures_part_0 = parts[0].getElementsByClass(music21.stream.Measure)
    key = measures_part_0[0][4]

    # estimate key of a song
    if not isinstance(key, music21.key.Key):
        key = song.analyze("key")

    print(key)


def dissonance_count(music_score):  # Returns a dictionary of the most common dissonances
    """
    Dissonances are described as follows:
    ‘Q’: Dissonant third quarter (special type of accented passing
    tone)
    ‘D’: Descending passing tone
    ‘R’: Ascending (“rising”) passing tone
    ‘L’: Lower neighbour
    ‘U’: Upper neighbour
    ‘S’: Suspension
    ‘F’: Fake suspension
    ‘f’: Diminished fake suspension
    ‘A’: Anticipation
    ‘C’: Nota cambiata
    ‘H’: Chanson idiom
    ‘E’: Echappée (escape tone)
    ‘-‘: Either no dissonance, or the part in question is not
    considered to be the dissonant note of the dissonance it’s in.
    :param music_score: a music score in the following formats: MIDI, Humdrum, xml.
    :return: dictionary of most common dissonances in the music piece
    """
    imported_score = Importer(music_score)
    dissonance = imported_score.get_data('dissonance')
    offset_dissonance = imported_score.get_data('dissonance')
    print(offset_dissonance)
    dissonance_dict = {}
    for dissonance_list in dissonance.values:
        for dissonance in dissonance_list:
            if dissonance is not "-":
                if dissonance not in dissonance_dict.keys():
                    dissonance_dict[dissonance] = 1
                else:
                    dissonance_dict[dissonance] += 1
    print(dissonance_dict)


def intervals_counter(music_score):
    imported_score = Importer(music_score)
    intervals_dict = {}
    hor_int_settings = {
        'quality': 'diatonic with quality',
        'simple or compound': 'compound',
        'directed': False
    }
    horizontal_intervals = imported_score.get_data('horizontal_interval', settings=hor_int_settings)
    freqs = imported_score.get_data('frequency', data=horizontal_intervals)
    for interval_type in freqs:
        for i, num_occurrences in enumerate(interval_type.index):
            intervals_dict[num_occurrences] = occurrences_sum(interval_type, i)
        for w in sorted(intervals_dict, key=intervals_dict.get, reverse=True):
            print(w, intervals_dict[w])


def melodic_pattern_no_rest_finder(music_score, ngrams=3):
    imported_score = Importer(music_score)
    pattern_dict = {}
    horizontal_settings = {
            'quality': 'diatonic with quality',
            'simple or compound': 'compound',
            'directed': True,
            'horiz_attach_later': True,
            'mp': False
        }
    ngram_settings = {
            'n': ngrams,
            'vertical': 'all',
            'brackets': False
        }

    horizontal_intervals = imported_score.get_data('horizontal_interval', settings=horizontal_settings)
    n_grams = imported_score.get_data('ngram', data=[horizontal_intervals], settings=ngram_settings)
    ngrams_occurrences = imported_score.get_data('frequency', data=n_grams)

    for occurrence in ngrams_occurrences:
        for i, y in enumerate(occurrence.values):
            pattern = occurrence.index[i]
            num_pattern_occurrences = occurrences_sum(occurrence, i)
            if 'Rest' not in pattern.split(' '):
                pattern_dict[pattern] = num_pattern_occurrences
        for w in sorted(pattern_dict, key=pattern_dict.get, reverse=True):
            print(w, pattern_dict[w])


def melodic_pattern_rest_finder(music_score, ngrams=3):
    imported_score = Importer(music_score)
    pattern_dict = {}
    horizontal_settings = {
        'quality': 'diatonic with quality',
        'simple or compound': 'compound',
        'directed': True,
        'horiz_attach_later': True,
        'mp': False
    }
    ngram_settings = {
        'n': ngrams,
        'vertical': 'all',
        'brackets': False
    }

    horizontal_intervals = imported_score.get_data('horizontal_interval', settings=horizontal_settings)
    n_grams = imported_score.get_data('ngram', data=[horizontal_intervals], settings=ngram_settings)
    ngrams_occurrences = imported_score.get_data('frequency', data=n_grams)

    for occurrence in ngrams_occurrences:
        for i, y in enumerate(occurrence.values):
            pattern = occurrence.index[i]
            num_pattern_occurrences = occurrences_sum(occurrence, i)
            pattern_dict[pattern] = num_pattern_occurrences
        for w in sorted(pattern_dict, key=pattern_dict.get, reverse=True):
            print(w, pattern_dict[w])


def rhythmic_pattern(music_score, ngrams=5):
    imported_score = Importer(music_score)
    pattern_dict = {}
    ngram_settings = {
                'n': ngrams,
                'vertical': 'all',
                'brackets': False
            }

    duration = imported_score.get_data('duration')
    n_grams = imported_score.get_data('ngram', data=[duration], settings=ngram_settings)
    ngrams_occurrences = imported_score.get_data('frequency', data=n_grams)
    for occurrence in ngrams_occurrences:
        for i, y in enumerate(occurrence.values):
            pattern = occurrence.index[i]
            num_pattern_occurrences = occurrences_sum(occurrence, i)
            pattern_dict[pattern] = num_pattern_occurrences
        for w in sorted(pattern_dict, key=pattern_dict.get, reverse=True):
            print(w, pattern_dict[w])


def identical_pattern(music_score, ngrams=9, consider_rests=False):

    score = music21.converter.parse(music_score)
    notes_list = []
    ngrams_list = list()
    ngrams_dict = {}
    instruments = instrument.partitionByInstrument(score)
    for inst in instruments:
        for note in inst.flat.notesAndRests:
            if consider_rests is True:
                notes_list.append(note.fullName)
            if consider_rests is False:
                if note.isNote:
                    notes_list.append(note.fullName)
        for i, note_listed in enumerate(notes_list):
            for num_ngrams in range(int(ngrams)):
                if i - ngrams + 1 > 0:
                    grams = notes_list[(i - ngrams) + num_ngrams]
                    ngrams_list.append(grams)
            if len(ngrams_list) != 0:
                if tuple(ngrams_list) not in ngrams_dict.keys():
                    ngrams_dict[tuple(ngrams_list)] = 1
                else:
                    ngrams_dict[tuple(ngrams_list)] += 1
            ngrams_list = []
        for w in sorted(ngrams_dict, key=ngrams_dict.get, reverse=True):
            if ngrams_dict[w] > 1 and len(w) is not 0:
                print(w, ngrams_dict[w], inst)
        notes_list = []


def max_identical_pattern(music_score, max_range=30, consider_rests=True):
    """
    Returns a the max length n-gram.
    The parameter max_range defines the max length n-gram we want to extract.
    """
    score = music21.converter.parse(music_score)
    notes_list = []
    ngrams_list = list()
    ngrams_dict = {}
    for note in score.flat.notesAndRests:
        if consider_rests is True:
            notes_list.append(note.fullName)
        if consider_rests is False:
            if note.isNote:
                notes_list.append(note.fullName)
    for i, note_listed in enumerate(notes_list):
        ngrams = max_range
        for num_ngrams in range(int(ngrams)):
            if i - ngrams + 1 > 0:
                grams = notes_list[(i - ngrams) + num_ngrams]
                ngrams_list.append(grams)
        if len(ngrams_list) != 0:
            if tuple(ngrams_list) not in ngrams_dict.keys():
                ngrams_dict[tuple(ngrams_list)] = 1
            else:
                ngrams_dict[tuple(ngrams_list)] += 1
        ngrams_list = []
    unique_values = set(ngrams_dict.values())
    if len(unique_values) <= 1:
        max_range -= 1
        max_identical_pattern(music_score, max_range)
    else:
        for w in sorted(ngrams_dict, key=ngrams_dict.get, reverse=True):
            if ngrams_dict[w] > 1 and len(w) is not 0:
                print(w, ngrams_dict[w], max_range)


if __name__ == "__main__":
    get_metadata("/Users/andrea/Documents/DHDK/Thesis/MIDI/essen/europa/deutschl/test/deut5146.krn")
    music_key(score_1)
    dissonance_count(score_2)
    intervals_counter(score_1)
    melodic_pattern_no_rest_finder(score_1, ngrams=5)
    melodic_pattern_rest_finder(score_1, ngrams=3)
    rhythmic_pattern(score_1, ngrams=3)
    identical_pattern(score_2, ngrams=3, consider_rests=True)
    max_identical_pattern(score_1, max_range=30, consider_rests=True)
