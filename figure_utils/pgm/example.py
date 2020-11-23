import daft
from matplotlib import rc

rc("font", family="serif", size=12)
rc("text", usetex=True)


pgm = daft.PGM()
pgm.add_node("cloudy", r"cloudy", 3, 3, aspect=1.8)
pgm.add_node("rain", r"rain", 2, 2, aspect=1.2)
pgm.add_node("sprinkler", r"sprinkler", 4, 2, aspect=2.1)
pgm.add_node("wet", r"grass wet", 3, 1, aspect=2.4, observed=True)
pgm.add_edge("cloudy", "rain", label=r"65\%", xoffset=-0.1, label_params={"rotation": 45})
pgm.add_edge("cloudy", "sprinkler", label=r"35\%", xoffset=0.1, label_params={"rotation": -45})
pgm.add_edge("rain", "wet")
pgm.add_edge("sprinkler", "wet")

pgm.render()
pgm.savefig("wordy.pdf")
pgm.savefig("wordy.png", dpi=150)
