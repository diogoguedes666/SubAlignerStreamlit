import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def preprocess_transfer_function(df):
    df['Frequency'] = pd.to_numeric(df['Frequency'], errors='coerce')
    df['Magnitude (dB)'] = pd.to_numeric(df['Magnitude (dB)'], errors='coerce')
    df['Phase (degrees)'] = pd.to_numeric(df['Phase (degrees)'], errors='coerce')
    df['Coherence'] = pd.to_numeric(df['Coherence'], errors='coerce')
    df.dropna(subset=['Frequency', 'Magnitude (dB)', 'Phase (degrees)', 'Coherence'], inplace=True)
    return df

def unwrap_phase(phase):
    return np.unwrap(np.radians(phase))

def calculate_best_delay(df1, df2, crossover_freq_hz, amp_diff_dB):
    # Determine frequency range based on amplitude difference range around the crossover frequency
    amp_diff_linear = 10 ** (amp_diff_dB / 20)  # Convert dB to linear scale
    crossover_amp1 = np.interp(crossover_freq_hz, df1['Frequency'], df1['Magnitude (dB)'])
    crossover_amp2 = np.interp(crossover_freq_hz, df2['Frequency'], df2['Magnitude (dB)'])

    lower_freq = crossover_freq_hz
    upper_freq = crossover_freq_hz

    # Find lower frequency boundary where amplitude is within amp_diff_dB range
    while lower_freq >= df1['Frequency'].min() and abs(crossover_amp1 - np.interp(lower_freq, df1['Frequency'], df1['Magnitude (dB)'])) <= amp_diff_dB:
        lower_freq -= 1

    # Find upper frequency boundary where amplitude is within amp_diff_dB range
    while upper_freq <= df1['Frequency'].max() and abs(crossover_amp1 - np.interp(upper_freq, df1['Frequency'], df1['Magnitude (dB)'])) <= amp_diff_dB:
        upper_freq += 1

    # Trim dataframes to the determined frequency range
    df1_range = df1[(df1['Frequency'] >= lower_freq) & (df1['Frequency'] <= upper_freq)]
    df2_range = df2[(df2['Frequency'] >= lower_freq) & (df2['Frequency'] <= upper_freq)]

    if df1_range.empty or df2_range.empty:
        return None, None

    phase1_unwrapped = unwrap_phase(df1_range['Phase (degrees)'])
    phase2_unwrapped = unwrap_phase(df2_range['Phase (degrees)'])

    freqs = df1_range['Frequency'].values

    # Check initial phase difference at the crossover frequency
    crossover_phase_diff = np.interp(crossover_freq_hz, df1_range['Frequency'], phase1_unwrapped) - \
                           np.interp(crossover_freq_hz, df2_range['Frequency'], phase2_unwrapped)
    initial_phase_diff = (crossover_phase_diff + np.pi) % (2 * np.pi) - np.pi

    # Initialize variables to find the best delay and polarity
    best_delay = 0
    best_polarity = 'normal'
    min_error = float('inf')

    # Evaluate both normal and inverted polarities
    for polarity in ['normal', 'invert']:
        if polarity == 'invert':
            phase2_unwrapped = unwrap_phase((df2_range['Phase (degrees)'] + 180) % 360)
        
        # Test delays from -20 ms to 20 ms
        delays = np.linspace(-20, 20, 400)
        for delay in delays:
            adjusted_phase2 = phase2_unwrapped - 2 * np.pi * freqs * delay / 1000

            # Align the phases to the closest multiple of 180 degrees
            phase_diff = (phase1_unwrapped - adjusted_phase2 + np.pi) % (2 * np.pi) - np.pi

            # Calculate the error as the sum of squared differences
            total_error = np.sum(phase_diff ** 2)
            
            if total_error < min_error:
                min_error = total_error
                best_delay = delay
                best_polarity = polarity

    return best_delay, best_polarity


def plot_transfer_function(df_list, crossover_freqs, coherence_tolerance, amp_diff_dB):
    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(10, 12))

    sum_complex_before = np.zeros(len(df_list[0]), dtype=np.complex128)
    sum_complex_after = np.zeros(len(df_list[0]), dtype=np.complex128)
    frequency = df_list[0]['Frequency'].values

    for idx, df in enumerate(df_list):
        df_filtered = df[df['Coherence'] >= coherence_tolerance]

        if df_filtered.empty:
            continue

        amplitude_linear_filtered = 10 ** (df_filtered['Magnitude (dB)'] / 20)
        phase_radians_filtered = unwrap_phase(df_filtered['Phase (degrees)'])
        complex_response_filtered = amplitude_linear_filtered * np.exp(1j * phase_radians_filtered)

        sum_complex_before += np.interp(frequency, df_filtered['Frequency'], complex_response_filtered, left=0, right=0)

        if idx > 0:
            delay_ms, polarity = calculate_best_delay(df_list[0], df, crossover_freqs[idx-1], amp_diff_dB)
            if polarity == 'invert':
                df_filtered['Phase (degrees)'] = (df_filtered['Phase (degrees)'] + 180) % 360
            df_filtered['Phase (degrees)'] = np.degrees(unwrap_phase(df_filtered['Phase (degrees)']) - 2 * np.pi * df_filtered['Frequency'] * delay_ms / 1000)

        amplitude_linear = 10 ** (df_filtered['Magnitude (dB)'] / 20)
        phase_radians = unwrap_phase(df_filtered['Phase (degrees)'])
        complex_response = amplitude_linear * np.exp(1j * phase_radians)

        sum_complex_after += np.interp(frequency, df_filtered['Frequency'], complex_response, left=0, right=0)
        ax_mag.plot(df_filtered['Frequency'], 20 * np.log10(amplitude_linear_filtered), label=f'Trace {idx + 1}')
        ax_phase.plot(df_filtered['Frequency'], df_filtered['Phase (degrees)'], label=f'Trace {idx + 1}')

    sum_amplitude_before = 20 * np.log10(np.abs(sum_complex_before))
    sum_amplitude_after = 20 * np.log10(np.abs(sum_complex_after))
    sum_phase_after = np.degrees(np.angle(sum_complex_after))

    ax_mag.plot(frequency, sum_amplitude_before, 'gray', linestyle='--', label='Sum Before Alignment')
    ax_mag.plot(frequency, sum_amplitude_after, 'r--', label='Sum After Alignment')
    ax_phase.plot(frequency, sum_phase_after, 'r--', label='Sum After Alignment')

    for crossover_freq in crossover_freqs:
        ax_mag.axvline(x=crossover_freq, color='r', linestyle='--', label=f'Crossover Frequency: {crossover_freq} Hz')
        ax_phase.axvline(x=crossover_freq, color='r', linestyle='--', label=f'Crossover Frequency: {crossover_freq} Hz')

    ax_mag.set_xscale('log')
    ax_mag.set_xlim(left=20)
    x_ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    ax_mag.set_xticks(x_ticks)
    ax_mag.set_xticklabels([str(tick) for tick in x_ticks])
    ax_mag.set_xlabel('Frequency (Hz)')
    ax_mag.set_ylabel('Magnitude (dB)')
    ax_mag.set_title('Magnitude Response')
    ax_mag.grid(True)
    ax_mag.legend()

    ax_phase.set_xscale('log')
    ax_phase.set_xlim(left=20)
    ax_phase.set_xticks(x_ticks)
    ax_phase.set_xticklabels([str(tick) for tick in x_ticks])
    ax_phase.set_ylim([-180, 180])
    ax_phase.axhline(y=0, color='k', linestyle='--')
    ax_phase.set_xlabel('Frequency (Hz)')
    ax_phase.set_ylabel('Phase (degrees)')
    ax_phase.set_title('Phase Response')
    ax_phase.grid(True)
    ax_phase.legend()

    st.pyplot(fig)

def main():
    st.title('Sub DelayCalculator')

    categories = ['subs', 'lowmids']
    file_dict = {category: None for category in categories}
    df_list = []
    crossover_freqs = []

    for idx, category in enumerate(categories):
        
        file_dict[category] = st.file_uploader(f"Upload {category.capitalize()} Transfer Function File", type=['csv', 'txt'])

        if file_dict[category] is not None:
            

            try:
                df = pd.read_csv(file_dict[category], delimiter='\t')
                df.columns = ['Frequency', 'Magnitude (dB)', 'Phase (degrees)', 'Coherence']
                df = preprocess_transfer_function(df)

                if not df.empty:
                    df_list.append(df)
                    if idx > 0:
                        crossover_text = f"Enter Crossover Frequency between {categories[idx-1].capitalize()} and {category.capitalize()} (Hz)"
                        crossover_freq = st.slider(crossover_text, min_value=20, max_value=500, value=160, step=1)
                        crossover_freqs.append(crossover_freq)

            except Exception as e:
                st.error(f"Error occurred while reading file: {e}")

    coherence_tolerance = st.slider("Coherence Tolerance", min_value=0.25, max_value=1.0, value=0.5, step=0.01)
    amp_diff_dB = st.slider("Isolation Zone dB Range", min_value=1, max_value=90
                        , value=16, step=1)

    if df_list and len(crossover_freqs) == len(df_list) - 1:
        st.header('Alignment Information')

        total_delay_ms = 0
        polarity_adjustments = []

        for df, crossover_freq in zip(df_list[1:], crossover_freqs):
            delay_ms, polarity = calculate_best_delay(df_list[0], df, crossover_freq, amp_diff_dB)
            if delay_ms is not None:
                total_delay_ms += delay_ms
                polarity_adjustments.append(polarity)

        if total_delay_ms < 0:
            displayed_delay = -total_delay_ms  # Make delay positive if calculated delay is negative
        else:
            displayed_delay = total_delay_ms

        st.markdown(f"**Total Delay for Alignment**: {displayed_delay:.2f} ms")
        st.markdown(f"**Polarity Adjustments**: {' '.join(polarity_adjustments)}")

        plot_transfer_function(df_list, crossover_freqs, coherence_tolerance, amp_diff_dB)

if __name__ == "__main__":
    main()