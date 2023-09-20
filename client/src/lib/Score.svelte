<script>
    import { onMount } from "svelte";
    import { darkMode } from "./stores";

    let isDarkMode;
    darkMode.subscribe((value) => {
        isDarkMode = value
    })

</script>

<div id="score-div">
  <h2 id="score-table-title" class:dark-mode={isDarkMode}>Score</h2>
  <div class="metric-div">
      <h4 class="metric-name" class:dark-mode={isDarkMode} title="This measures how long it takes for a support agent to respond to a customer's query. Rapid responses can significantly improve customer satisfaction.">Response Time</h4>
      <h4 class="metric-score" class:dark-mode={isDarkMode} id="response-time-score">0</h4>
  </div>
  <div class="metric-div">
      <h4 class="metric-name" class:dark-mode={isDarkMode} title="This is usually measured through surveys sent after the resolution of a ticket, asking the customer to rate their overall satisfaction with the service received.">Customer Satisfaction</h4>
      <h4 class="metric-score" class:dark-mode={isDarkMode} id="csat-score">0</h4>
  </div>
  <div class="metric-div">
      <h4 class="metric-name" class:dark-mode={isDarkMode} title="This metric gauges the likelihood that the customer would recommend the company (or specifically its customer service) to others. It's a measure of customer loyalty and can be influenced by the quality of support received.">Net Promoter</h4>
      <h4 class="metric-score" class:dark-mode={isDarkMode} id="nps-score">0</h4>
  </div>
  <div class="metric-div">
      <h4 class="metric-name" class:dark-mode={isDarkMode} title="This measures how much effort a customer has to exert to get their issue resolved.">Customer Effort</h4>
      <h4 class="metric-score" class:dark-mode={isDarkMode} id="ces-score">0</h4>
  </div>
  <div class="metric-div">
      <h4 class="metric-name" class:dark-mode={isDarkMode} title="This is often assessed through customer feedback surveys and can include factors such as the agent's knowledge, communication skills, and ability to solve the problem effectively.">Quality of Response</h4>
      <h4 class="metric-score" class:dark-mode={isDarkMode} id="response-quality-score">0</h4>
  </div>
  <div class="metric-div">
      <h3 class="metric-name" class:dark-mode={isDarkMode}>Total</h3>
      <h2 class="metric-score" class:dark-mode={isDarkMode} id="total-score">0</h2>
  </div>
</div>
