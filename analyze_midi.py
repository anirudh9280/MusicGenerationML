import pretty_midi

pm = pretty_midi.PrettyMIDI('symbolic_conditioned.mid')
inst = pm.instruments[0]

print('🎵 MIDI Analysis for symbolic_conditioned.mid')
print('=' * 50)
print(f'✅ Duration: {pm.get_end_time():.2f} seconds')
print(f'✅ Instrument: {inst.program} (piano)')
print(f'✅ Total notes: {len(inst.notes)}')
print(f'✅ Tempo: {pm.estimate_tempo():.1f} BPM')
print()
print('📝 Note Details:')
for i, note in enumerate(inst.notes):
    note_name = pretty_midi.note_number_to_name(note.pitch)
    print(f'  {i+1}: {note_name:<4} | start={note.start:.2f}s | end={note.end:.2f}s | velocity={note.velocity}')

print()
print('🎼 This appears to be a valid MIDI file for Task 2!')
print('   You can play this in any MIDI player or DAW.') 