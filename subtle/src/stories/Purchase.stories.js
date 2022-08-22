export default {
  title: 'Example/PurchaseButton',
  argTypes: {
   
  },
};

// More on component templates: https://storybook.js.org/docs/html/writing-stories/introduction#using-args
const Template = ({ label, ...args }) => {

  
  // You can either use a function to create DOM elements or use a plain html string!
  return `<div class="purchase" onclick="this.classList.add('added')">${label}</div>`;
};

export const Primary = Template.bind({});
// More on args: https://storybook.js.org/docs/html/writing-stories/args
Primary.args = {
  primary: true,
  label: 'Button',
};
