from music21 import *
from pychord import note_to_chord
import re

b = corpus.parse('bwv1.6')
my_file = "url_to_symbolic_file"


bChords = b.chordify()
# bChords.show()
print(bChords.secondsMap)

for thisChord in bChords.recurse().getElementsByClass('Chord'):
    # if thisChord.isDominantSeventh():
    # print(thisChord.measureNumber, thisChord.beatStr, thisChord)
    final_chord = thisChord.pitchedCommonName
    #print(clean_chord)
    print(final_chord, thisChord, thisChord.offset, thisChord.measureNumber)

b.insert(0, bChords)

for c in bChords.recurse().getElementsByClass('Chord'):
    c.closedPosition(forceOctave=4, inPlace=True)

for c in bChords.recurse().getElementsByClass('Chord'):
    rn = roman.romanNumeralFromChord(c, key.Key('A'))
    c.addLyric(str(rn.figure))

# b.show()

for c in bChords.measures(0,2).flat:
    if 'Chord' not in c.classes:
        continue
    print(c.lyric, end=' ')
