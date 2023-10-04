import os
import midi2audio
from midi2audio import FluidSynth
import subprocess


class MusicFileConverter:
    def __init__(self, sound_font="sound_font.sf2", sample_rate=44100, abc2xml_path_file="abc2xml.py", xml2abc_path_file="xml2abc.py"):
        self._supported_formats = {
            "abc": [".abc", ".txt"],
            "midi": [".mid", ".midi"],
            "xml": [".xml"],
            "wav": [".wav"],
        }
        self.sound_font = sound_font
        self.sample_rate = sample_rate
        self.abc2xml_path_file = abc2xml_path_file
        self.xml2abc_path_file = xml2abc_path_file

        # Initialize FluidSynth
        FluidSynth(self.sound_font)
        FluidSynth(sample_rate=self.sample_rate)

    def _check_format(self, filename, file_format):
        _, ext = os.path.splitext(filename)
        return ext.lower() in self._supported_formats.get(file_format, [])

    def midi_to_wav(self, midi_file, wav_file):
        if not self._check_format(midi_file, "midi") or not self._check_format(
            wav_file, "wav"
        ):
            print("Invalid file format!")
            return None

        FluidSynth().midi_to_audio(midi_file, wav_file)

        return wav_file

    def midi_to_abc(self, midi_file, abc_file):
        if not self._check_format(midi_file, "midi") or not self._check_format(
            abc_file, "abc"
        ):
            print("Invalid file format!")
            return None

        # Use midi2abc to convert midi to abc
        subprocess.run(["midi2abc", midi_file, "-o", abc_file])

        return abc_file

    def abc_to_midi(self, abc_file, midi_file):
        if not self._check_format(abc_file, "abc") or not self._check_format(
            midi_file, "midi"
        ):
            print("Invalid file format!")
            return None

        # Use abcmidi to convert abc to midi
        subprocess.run(["abc2midi", abc_file, "-o", midi_file])

        return midi_file

    def abc_to_xml(self, abc_file, xml_folder):
        if not self._check_format(abc_file, "abc"):
            print("Invalid file format!")
            return None
        
        #  Make subprocess call to abc2xml.py 
        subprocess.run(["python", self.abc2xml_path_file, "-o", xml_folder, abc_file])

        return abc_file.replace(".abc", ".xml")

    def xml_to_abc(self, xml_file, abc_folder):
        if not self._check_format(xml_file, "xml"):
            print("Invalid file format!")
            return None

        # Implement the conversion from xml to abc using the external library
        subprocess.run(["python", self.xml2abc_path_file, "-o", abc_folder, xml_file])

        # Return the path of the abc file
        path_abc = self.get_parent_folder(xml_file) + "/" + self.get_file_name(xml_file).replace(".xml", ".abc")

        return path_abc

    def xml_to_midi(self, xml_file, midi_file):
        if not self._check_format(xml_file, "xml") or not self._check_format(
            midi_file, "midi"
        ):
            print("Invalid file format!")
            return

        # Implement the conversion from xml to midi using the external library
        # First convert xml to abc
        abc_file = self.xml_to_abc(xml_file, self.get_parent_folder(xml_file))
        # Then convert abc to midi
        midi_file = self.abc_to_midi(abc_file, midi_file)

        # Delete abc file
        os.remove(abc_file)

        return midi_file


    def xml_to_wav(self, xml_file, wav_file):
        if not self._check_format(xml_file, "xml") or not self._check_format(
            wav_file, "wav"
        ):
            print("Invalid file format!")
            return

        # Implement the conversion from xml to wav using the external library
        # First convert xml to midi
        midi_file = self.xml_to_midi(xml_file, self.get_file_name(xml_file).replace(".xml", ".mid"))

        # Then convert midi to wav
        wav_file = self.midi_to_wav(midi_file, wav_file)

        # Delete midi file
        os.remove(midi_file)

        return wav_file


    def wav_to_midi(self, wav_file, midi_file):
        if not self._check_format(wav_file, "wav") or not self._check_format(
            midi_file, "midi"
        ):
            print("Invalid file format!")
            return

        # Implement the conversion from wav to midi using the external library

        # Launch exception not implemented yet
        raise NotImplementedError("THIS REQUIRES SAGESHEET ENDPOINT- Coming soon!")

    # function which extrat the name of the file from the path
    def get_file_name(self, file_path):
        return os.path.basename(file_path)
    
    # function which extract the path of the parent folder of the file path
    def get_parent_folder(self, file_path):
        return os.path.dirname(file_path)