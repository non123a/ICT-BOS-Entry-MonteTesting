# ICT-BOS-Entry-MonteTesting


v2.3 adv is a backtest script contact WFO (walk forwad opitimze) to more accrucy of the strategy with the advance version it to sweep between parameter to see if the strategy work when the parameter change, so not just 1 parameter work then the rest is falling then it mean that strategy caought noise rather accruracy data

v2.3 adv2 is add with the monte to sweep between data to see if the strategy survive the random switch between data, this is make so we know if how much is the drawdown on the strategy and to see if it survive randomnese of the market.

v2.3 adv2.1 is add more feature to the monte to make it with bootstrap to Simulates unseen future trades More realistic than simple shuffle, Add Realism ( slippege and spread)

forward-test-adv-1 is a forward test for the data from 09-25 it contact all the real senario like slippage spread, misesed order, but it do not force close position after newyork end.

forward-test-adv-2 is adding ATR 14 and filtering out weak BOS, so not as much trade taking as before but it ment to improve wr of the strategy, which is the old one is around 44-54% depend on the parameter.